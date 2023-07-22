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
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline static void MatrixTransposedTimesMatrix(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue cols, const XT *RESTRICT x, const YT *RESTRICT y,
    SubscriptValue n) {
  using ResultType = CppTypeFor<RCAT, RKIND>;

  std::memset(product, 0, rows * cols * sizeof *product);
  for (SubscriptValue j{0}; j < cols; ++j) {
    for (SubscriptValue i{0}; i < rows; ++i) {
      for (SubscriptValue k{0}; k < n; ++k) {
        ResultType x_ki = static_cast<ResultType>(x[i * n + k]);
        ResultType y_kj = static_cast<ResultType>(y[j * n + k]);
        product[j * rows + i] += x_ki * y_kj;
      }
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
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline static void MatrixTransposedTimesVector(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue n, const XT *RESTRICT x, const YT *RESTRICT y) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, rows * sizeof *product);
  for (SubscriptValue i{0}; i < rows; ++i) {
    for (SubscriptValue k{0}; k < n; ++k) {
      ResultType x_ki = static_cast<ResultType>(x[i * n + k]);
      ResultType y_k = static_cast<ResultType>(y[k]);
      product[i] += x_ki * y_k;
    }
  }
}

// Implements an instance of MATMUL for given argument types.
template <bool IS_ALLOCATING, TypeCategory RCAT, int RKIND, typename XT,
    typename YT>
inline static void DoMatmulTranspose(
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
    if (x.IsContiguous() && y.IsContiguous() &&
        (IS_ALLOCATING || result.IsContiguous())) {
      // Contiguous numeric matrices
      if (resRank == 2) { // M*M -> M
        MatrixTransposedTimesMatrix<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), rows, cols,
            x.OffsetElement<XT>(), y.OffsetElement<YT>(), n);
        return;
      }
      if (xRank == 2) { // M*V -> V
        MatrixTransposedTimesVector<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), rows, n,
            x.OffsetElement<XT>(), y.OffsetElement<YT>());
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

// Maps the dynamic type information from the arguments' descriptors
// to the right instantiation of DoMatmul() for valid combinations of
// types.
template <bool IS_ALLOCATING> struct MatmulTranspose {
  using ResultDescriptor =
      std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor>;
  template <TypeCategory XCAT, int XKIND> struct MM1 {
    template <TypeCategory YCAT, int YKIND> struct MM2 {
      void operator()(ResultDescriptor &result, const Descriptor &x,
          const Descriptor &y, Terminator &terminator) const {
        if constexpr (constexpr auto resultType{
                          GetResultType(XCAT, XKIND, YCAT, YKIND)}) {
          if constexpr (Fortran::common::IsNumericTypeCategory(
                            resultType->first) ||
              resultType->first == TypeCategory::Logical) {
            return DoMatmulTranspose<IS_ALLOCATING, resultType->first,
                resultType->second, CppTypeFor<XCAT, XKIND>,
                CppTypeFor<YCAT, YKIND>>(result, x, y, terminator);
          }
        }
        terminator.Crash("MATMUL-TRANSPOSE: bad operand types (%d(%d), %d(%d))",
            static_cast<int>(XCAT), XKIND, static_cast<int>(YCAT), YKIND);
      }
    };
    void operator()(ResultDescriptor &result, const Descriptor &x,
        const Descriptor &y, Terminator &terminator, TypeCategory yCat,
        int yKind) const {
      ApplyType<MM2, void>(yCat, yKind, terminator, result, x, y, terminator);
    }
  };
  void operator()(ResultDescriptor &result, const Descriptor &x,
      const Descriptor &y, const char *sourceFile, int line) const {
    Terminator terminator{sourceFile, line};
    auto xCatKind{x.type().GetCategoryAndKind()};
    auto yCatKind{y.type().GetCategoryAndKind()};
    RUNTIME_CHECK(terminator, xCatKind.has_value() && yCatKind.has_value());
    ApplyType<MM1, void>(xCatKind->first, xCatKind->second, terminator, result,
        x, y, terminator, yCatKind->first, yCatKind->second);
  }
};
} // namespace

namespace Fortran::runtime {
extern "C" {
void RTNAME(MatmulTranspose)(Descriptor &result, const Descriptor &x,
    const Descriptor &y, const char *sourceFile, int line) {
  MatmulTranspose<true>{}(result, x, y, sourceFile, line);
}
void RTNAME(MatmulTransposeDirect)(const Descriptor &result,
    const Descriptor &x, const Descriptor &y, const char *sourceFile,
    int line) {
  MatmulTranspose<false>{}(result, x, y, sourceFile, line);
}
} // extern "C"
} // namespace Fortran::runtime
