//===-- runtime/matmul-transpose.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements a fused matmul-transpose operation
//
// There are two main entry points; one establishes a descriptor for the
// result and allocates it, and the other expects a result descriptor that
// points to existing storage.
//
// This implementation must handle all combinations of numeric types and
// kinds (100 - 165 cases depending on the target), plus all combinations
// of logical kinds (16).  A single template undergoes many instantiations
// to cover all of the valid possibilities.
//
// The usefulness of this optimization should be reviewed once Matmul is swapped
// to use the faster BLAS routines.

#include "flang/Runtime/matmul-transpose.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Common/optional.h"
#include "flang/Runtime/c-or-cpp.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include <cstring>

namespace {
using namespace Fortran::runtime;

// Contiguous numeric TRANSPOSE(matrix)*matrix multiplication
//   TRANSPOSE(matrix(n, rows)) * matrix(n,cols) ->
//             matrix(rows, n)  * matrix(n,cols) -> matrix(rows,cols)
// The transpose is implemented by swapping the indices of accesses into the LHS
//
// Straightforward algorithm:
//   DO 1 I = 1, NROWS
//    DO 1 J = 1, NCOLS
//     RES(I,J) = 0
//     DO 1 K = 1, N
//   1  RES(I,J) = RES(I,J) + X(K,I)*Y(K,J)
//
// With loop distribution and transposition to avoid the inner sum
// reduction and to avoid non-unit strides:
//   DO 1 I = 1, NROWS
//    DO 1 J = 1, NCOLS
//   1 RES(I,J) = 0
//   DO 2 J = 1, NCOLS
//    DO 2 I = 1, NROWS
//     DO 2 K = 1, N
//   2  RES(I,J) = RES(I,J) + X(K,I)*Y(K,J) ! loop-invariant last term
template <TypeCategory RCAT, int RKIND, typename XT, typename YT,
    bool X_HAS_STRIDED_COLUMNS, bool Y_HAS_STRIDED_COLUMNS>
inline static RT_API_ATTRS void MatrixTransposedTimesMatrix(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue cols, const XT *RESTRICT x, const YT *RESTRICT y,
    SubscriptValue n, std::size_t xColumnByteStride = 0,
    std::size_t yColumnByteStride = 0) {
  using ResultType = CppTypeFor<RCAT, RKIND>;

  std::memset(product, 0, rows * cols * sizeof *product);
  for (SubscriptValue j{0}; j < cols; ++j) {
    for (SubscriptValue i{0}; i < rows; ++i) {
      for (SubscriptValue k{0}; k < n; ++k) {
        ResultType x_ki;
        if constexpr (!X_HAS_STRIDED_COLUMNS) {
          x_ki = static_cast<ResultType>(x[i * n + k]);
        } else {
          x_ki = static_cast<ResultType>(reinterpret_cast<const XT *>(
              reinterpret_cast<const char *>(x) + i * xColumnByteStride)[k]);
        }
        ResultType y_kj;
        if constexpr (!Y_HAS_STRIDED_COLUMNS) {
          y_kj = static_cast<ResultType>(y[j * n + k]);
        } else {
          y_kj = static_cast<ResultType>(reinterpret_cast<const YT *>(
              reinterpret_cast<const char *>(y) + j * yColumnByteStride)[k]);
        }
        product[j * rows + i] += x_ki * y_kj;
      }
    }
  }
}

template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline static RT_API_ATTRS void MatrixTransposedTimesMatrixHelper(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue cols, const XT *RESTRICT x, const YT *RESTRICT y,
    SubscriptValue n, Fortran::common::optional<std::size_t> xColumnByteStride,
    Fortran::common::optional<std::size_t> yColumnByteStride) {
  if (!xColumnByteStride) {
    if (!yColumnByteStride) {
      MatrixTransposedTimesMatrix<RCAT, RKIND, XT, YT, false, false>(
          product, rows, cols, x, y, n);
    } else {
      MatrixTransposedTimesMatrix<RCAT, RKIND, XT, YT, false, true>(
          product, rows, cols, x, y, n, 0, *yColumnByteStride);
    }
  } else {
    if (!yColumnByteStride) {
      MatrixTransposedTimesMatrix<RCAT, RKIND, XT, YT, true, false>(
          product, rows, cols, x, y, n, *xColumnByteStride);
    } else {
      MatrixTransposedTimesMatrix<RCAT, RKIND, XT, YT, true, true>(
          product, rows, cols, x, y, n, *xColumnByteStride, *yColumnByteStride);
    }
  }
}

// Contiguous numeric matrix*vector multiplication
//   matrix(rows,n) * column vector(n) -> column vector(rows)
// Straightforward algorithm:
//   DO 1 I = 1, NROWS
//    RES(I) = 0
//    DO 1 K = 1, N
//   1 RES(I) = RES(I) + X(K,I)*Y(K)
// With loop distribution and transposition to avoid the inner
// sum reduction and to avoid non-unit strides:
//   DO 1 I = 1, NROWS
//   1 RES(I) = 0
//   DO 2 I = 1, NROWS
//    DO 2 K = 1, N
//   2 RES(I) = RES(I) + X(K,I)*Y(K)
template <TypeCategory RCAT, int RKIND, typename XT, typename YT,
    bool X_HAS_STRIDED_COLUMNS>
inline static RT_API_ATTRS void MatrixTransposedTimesVector(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue n, const XT *RESTRICT x, const YT *RESTRICT y,
    std::size_t xColumnByteStride = 0) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, rows * sizeof *product);
  for (SubscriptValue i{0}; i < rows; ++i) {
    for (SubscriptValue k{0}; k < n; ++k) {
      ResultType x_ki;
      if constexpr (!X_HAS_STRIDED_COLUMNS) {
        x_ki = static_cast<ResultType>(x[i * n + k]);
      } else {
        x_ki = static_cast<ResultType>(reinterpret_cast<const XT *>(
            reinterpret_cast<const char *>(x) + i * xColumnByteStride)[k]);
      }
      ResultType y_k = static_cast<ResultType>(y[k]);
      product[i] += x_ki * y_k;
    }
  }
}

template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline static RT_API_ATTRS void MatrixTransposedTimesVectorHelper(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue n, const XT *RESTRICT x, const YT *RESTRICT y,
    Fortran::common::optional<std::size_t> xColumnByteStride) {
  if (!xColumnByteStride) {
    MatrixTransposedTimesVector<RCAT, RKIND, XT, YT, false>(
        product, rows, n, x, y);
  } else {
    MatrixTransposedTimesVector<RCAT, RKIND, XT, YT, true>(
        product, rows, n, x, y, *xColumnByteStride);
  }
}

// Implements an instance of MATMUL for given argument types.
template <bool IS_ALLOCATING, TypeCategory RCAT, int RKIND, typename XT,
    typename YT>
inline static RT_API_ATTRS void DoMatmulTranspose(
    std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor> &result,
    const Descriptor &x, const Descriptor &y, Terminator &terminator) {
  int xRank{x.rank()};
  int yRank{y.rank()};
  int resRank{xRank + yRank - 2};
  if (xRank * yRank != 2 * resRank) {
    terminator.Crash(
        "MATMUL-TRANSPOSE: bad argument ranks (%d * %d)", xRank, yRank);
  }
  SubscriptValue extent[2]{x.GetDimension(1).Extent(),
      resRank == 2 ? y.GetDimension(1).Extent() : 0};
  if constexpr (IS_ALLOCATING) {
    result.Establish(
        RCAT, RKIND, nullptr, resRank, extent, CFI_attribute_allocatable);
    for (int j{0}; j < resRank; ++j) {
      result.GetDimension(j).SetBounds(1, extent[j]);
    }
    if (int stat{result.Allocate()}) {
      terminator.Crash(
          "MATMUL-TRANSPOSE: could not allocate memory for result; STAT=%d",
          stat);
    }
  } else {
    RUNTIME_CHECK(terminator, resRank == result.rank());
    RUNTIME_CHECK(
        terminator, result.ElementBytes() == static_cast<std::size_t>(RKIND));
    RUNTIME_CHECK(terminator, result.GetDimension(0).Extent() == extent[0]);
    RUNTIME_CHECK(terminator,
        resRank == 1 || result.GetDimension(1).Extent() == extent[1]);
  }
  SubscriptValue n{x.GetDimension(0).Extent()};
  if (n != y.GetDimension(0).Extent()) {
    terminator.Crash(
        "MATMUL-TRANSPOSE: unacceptable operand shapes (%jdx%jd, %jdx%jd)",
        static_cast<std::intmax_t>(x.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(x.GetDimension(1).Extent()),
        static_cast<std::intmax_t>(y.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(y.GetDimension(1).Extent()));
  }
  using WriteResult =
      CppTypeFor<RCAT == TypeCategory::Logical ? TypeCategory::Integer : RCAT,
          RKIND>;
  const SubscriptValue rows{extent[0]};
  const SubscriptValue cols{extent[1]};
  if constexpr (RCAT != TypeCategory::Logical) {
    if (x.IsContiguous(1) && y.IsContiguous(1) &&
        (IS_ALLOCATING || result.IsContiguous())) {
      // Contiguous numeric matrices (maybe with columns
      // separated by a stride).
      Fortran::common::optional<std::size_t> xColumnByteStride;
      if (!x.IsContiguous()) {
        // X's columns are strided.
        SubscriptValue xAt[2]{};
        x.GetLowerBounds(xAt);
        xAt[1]++;
        xColumnByteStride = x.SubscriptsToByteOffset(xAt);
      }
      Fortran::common::optional<std::size_t> yColumnByteStride;
      if (!y.IsContiguous()) {
        // Y's columns are strided.
        SubscriptValue yAt[2]{};
        y.GetLowerBounds(yAt);
        yAt[1]++;
        yColumnByteStride = y.SubscriptsToByteOffset(yAt);
      }
      if (resRank == 2) { // M*M -> M
        // TODO: use BLAS-3 GEMM for supported types.
        MatrixTransposedTimesMatrixHelper<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), rows, cols,
            x.OffsetElement<XT>(), y.OffsetElement<YT>(), n, xColumnByteStride,
            yColumnByteStride);
        return;
      }
      if (xRank == 2) { // M*V -> V
        // TODO: use BLAS-2 GEMM for supported types.
        MatrixTransposedTimesVectorHelper<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), rows, n,
            x.OffsetElement<XT>(), y.OffsetElement<YT>(), xColumnByteStride);
        return;
      }
      // else V*M -> V (not allowed because TRANSPOSE() is only defined for rank
      // 1 matrices
      terminator.Crash(
          "MATMUL-TRANSPOSE: unacceptable operand shapes (%jdx%jd, %jdx%jd)",
          static_cast<std::intmax_t>(x.GetDimension(0).Extent()),
          static_cast<std::intmax_t>(n),
          static_cast<std::intmax_t>(y.GetDimension(0).Extent()),
          static_cast<std::intmax_t>(y.GetDimension(1).Extent()));
      return;
    }
  }
  // General algorithms for LOGICAL and noncontiguity
  SubscriptValue xLB[2], yLB[2], resLB[2];
  x.GetLowerBounds(xLB);
  y.GetLowerBounds(yLB);
  result.GetLowerBounds(resLB);
  using ResultType = CppTypeFor<RCAT, RKIND>;
  if (resRank == 2) { // M*M -> M
    for (SubscriptValue i{0}; i < rows; ++i) {
      for (SubscriptValue j{0}; j < cols; ++j) {
        ResultType res_ij;
        if constexpr (RCAT == TypeCategory::Logical) {
          res_ij = false;
        } else {
          res_ij = 0;
        }

        for (SubscriptValue k{0}; k < n; ++k) {
          SubscriptValue xAt[2]{k + xLB[0], i + xLB[1]};
          SubscriptValue yAt[2]{k + yLB[0], j + yLB[1]};
          if constexpr (RCAT == TypeCategory::Logical) {
            ResultType x_ki = IsLogicalElementTrue(x, xAt);
            ResultType y_kj = IsLogicalElementTrue(y, yAt);
            res_ij = res_ij || (x_ki && y_kj);
          } else {
            ResultType x_ki = static_cast<ResultType>(*x.Element<XT>(xAt));
            ResultType y_kj = static_cast<ResultType>(*y.Element<YT>(yAt));
            res_ij += x_ki * y_kj;
          }
        }
        SubscriptValue resAt[2]{i + resLB[0], j + resLB[1]};
        *result.template Element<WriteResult>(resAt) = res_ij;
      }
    }
  } else if (xRank == 2) { // M*V -> V
    for (SubscriptValue i{0}; i < rows; ++i) {
      ResultType res_i;
      if constexpr (RCAT == TypeCategory::Logical) {
        res_i = false;
      } else {
        res_i = 0;
      }

      for (SubscriptValue k{0}; k < n; ++k) {
        SubscriptValue xAt[2]{k + xLB[0], i + xLB[1]};
        SubscriptValue yAt[1]{k + yLB[0]};
        if constexpr (RCAT == TypeCategory::Logical) {
          ResultType x_ki = IsLogicalElementTrue(x, xAt);
          ResultType y_k = IsLogicalElementTrue(y, yAt);
          res_i = res_i || (x_ki && y_k);
        } else {
          ResultType x_ki = static_cast<ResultType>(*x.Element<XT>(xAt));
          ResultType y_k = static_cast<ResultType>(*y.Element<YT>(yAt));
          res_i += x_ki * y_k;
        }
      }
      SubscriptValue resAt[1]{i + resLB[0]};
      *result.template Element<WriteResult>(resAt) = res_i;
    }
  } else { // V*M -> V
    // TRANSPOSE(V) not allowed by fortran standard
    terminator.Crash(
        "MATMUL-TRANSPOSE: unacceptable operand shapes (%jdx%jd, %jdx%jd)",
        static_cast<std::intmax_t>(x.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(n),
        static_cast<std::intmax_t>(y.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(y.GetDimension(1).Extent()));
  }
}

template <bool IS_ALLOCATING, TypeCategory XCAT, int XKIND, TypeCategory YCAT,
    int YKIND>
struct MatmulTransposeHelper {
  using ResultDescriptor =
      std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor>;
  RT_API_ATTRS void operator()(ResultDescriptor &result, const Descriptor &x,
      const Descriptor &y, const char *sourceFile, int line) const {
    Terminator terminator{sourceFile, line};
    auto xCatKind{x.type().GetCategoryAndKind()};
    auto yCatKind{y.type().GetCategoryAndKind()};
    RUNTIME_CHECK(terminator, xCatKind.has_value() && yCatKind.has_value());
    RUNTIME_CHECK(terminator, xCatKind->first == XCAT);
    RUNTIME_CHECK(terminator, yCatKind->first == YCAT);
    if constexpr (constexpr auto resultType{
                      GetResultType(XCAT, XKIND, YCAT, YKIND)}) {
      return DoMatmulTranspose<IS_ALLOCATING, resultType->first,
          resultType->second, CppTypeFor<XCAT, XKIND>, CppTypeFor<YCAT, YKIND>>(
          result, x, y, terminator);
    }
    terminator.Crash("MATMUL-TRANSPOSE: bad operand types (%d(%d), %d(%d))",
        static_cast<int>(XCAT), XKIND, static_cast<int>(YCAT), YKIND);
  }
};
} // namespace

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

#define MATMUL_INSTANCE(XCAT, XKIND, YCAT, YKIND) \
  void RTDEF(MatmulTranspose##XCAT##XKIND##YCAT##YKIND)(Descriptor & result, \
      const Descriptor &x, const Descriptor &y, const char *sourceFile, \
      int line) { \
    MatmulTransposeHelper<true, TypeCategory::XCAT, XKIND, TypeCategory::YCAT, \
        YKIND>{}(result, x, y, sourceFile, line); \
  }

#define MATMUL_DIRECT_INSTANCE(XCAT, XKIND, YCAT, YKIND) \
  void RTDEF(MatmulTransposeDirect##XCAT##XKIND##YCAT##YKIND)( \
      Descriptor & result, const Descriptor &x, const Descriptor &y, \
      const char *sourceFile, int line) { \
    MatmulTransposeHelper<false, TypeCategory::XCAT, XKIND, \
        TypeCategory::YCAT, YKIND>{}(result, x, y, sourceFile, line); \
  }

#define MATMUL_FORCE_ALL_TYPES 0

#include "flang/Runtime/matmul-instances.inc"

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
