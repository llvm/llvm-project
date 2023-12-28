//===-- runtime/matmul.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements all forms of MATMUL (Fortran 2018 16.9.124)
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
// Places where BLAS routines could be called are marked as TODO items.

#include "flang/Runtime/matmul.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/c-or-cpp.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include <cstring>

namespace Fortran::runtime {

// Suppress the warnings about calling __host__-only std::complex operators,
// defined in C++ STD header files, from __device__ code.
RT_DIAG_PUSH
RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN

// General accumulator for any type and stride; this is not used for
// contiguous numeric cases.
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
class Accumulator {
public:
  using Result = AccumulationType<RCAT, RKIND>;
  RT_API_ATTRS Accumulator(const Descriptor &x, const Descriptor &y)
      : x_{x}, y_{y} {}
  RT_API_ATTRS void Accumulate(
      const SubscriptValue xAt[], const SubscriptValue yAt[]) {
    if constexpr (RCAT == TypeCategory::Logical) {
      sum_ = sum_ ||
          (IsLogicalElementTrue(x_, xAt) && IsLogicalElementTrue(y_, yAt));
    } else {
      sum_ += static_cast<Result>(*x_.Element<XT>(xAt)) *
          static_cast<Result>(*y_.Element<YT>(yAt));
    }
  }
  RT_API_ATTRS Result GetResult() const { return sum_; }

private:
  const Descriptor &x_, &y_;
  Result sum_{};
};

// Contiguous numeric matrix*matrix multiplication
//   matrix(rows,n) * matrix(n,cols) -> matrix(rows,cols)
// Straightforward algorithm:
//   DO 1 I = 1, NROWS
//    DO 1 J = 1, NCOLS
//     RES(I,J) = 0
//     DO 1 K = 1, N
//   1  RES(I,J) = RES(I,J) + X(I,K)*Y(K,J)
// With loop distribution and transposition to avoid the inner sum
// reduction and to avoid non-unit strides:
//   DO 1 I = 1, NROWS
//    DO 1 J = 1, NCOLS
//   1 RES(I,J) = 0
//   DO 2 K = 1, N
//    DO 2 J = 1, NCOLS
//     DO 2 I = 1, NROWS
//   2  RES(I,J) = RES(I,J) + X(I,K)*Y(K,J) ! loop-invariant last term
template <TypeCategory RCAT, int RKIND, typename XT, typename YT,
    bool X_HAS_STRIDED_COLUMNS, bool Y_HAS_STRIDED_COLUMNS>
inline RT_API_ATTRS void MatrixTimesMatrix(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue cols, const XT *RESTRICT x, const YT *RESTRICT y,
    SubscriptValue n, std::size_t xColumnByteStride = 0,
    std::size_t yColumnByteStride = 0) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, rows * cols * sizeof *product);
  const XT *RESTRICT xp0{x};
  for (SubscriptValue k{0}; k < n; ++k) {
    ResultType *RESTRICT p{product};
    for (SubscriptValue j{0}; j < cols; ++j) {
      const XT *RESTRICT xp{xp0};
      ResultType yv;
      if constexpr (!Y_HAS_STRIDED_COLUMNS) {
        yv = static_cast<ResultType>(y[k + j * n]);
      } else {
        yv = static_cast<ResultType>(reinterpret_cast<const YT *>(
            reinterpret_cast<const char *>(y) + j * yColumnByteStride)[k]);
      }
      for (SubscriptValue i{0}; i < rows; ++i) {
        *p++ += static_cast<ResultType>(*xp++) * yv;
      }
    }
    if constexpr (!X_HAS_STRIDED_COLUMNS) {
      xp0 += rows;
    } else {
      xp0 = reinterpret_cast<const XT *>(
          reinterpret_cast<const char *>(xp0) + xColumnByteStride);
    }
  }
}

RT_DIAG_POP

template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline RT_API_ATTRS void MatrixTimesMatrixHelper(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue cols, const XT *RESTRICT x, const YT *RESTRICT y,
    SubscriptValue n, std::optional<std::size_t> xColumnByteStride,
    std::optional<std::size_t> yColumnByteStride) {
  if (!xColumnByteStride) {
    if (!yColumnByteStride) {
      MatrixTimesMatrix<RCAT, RKIND, XT, YT, false, false>(
          product, rows, cols, x, y, n);
    } else {
      MatrixTimesMatrix<RCAT, RKIND, XT, YT, false, true>(
          product, rows, cols, x, y, n, 0, *yColumnByteStride);
    }
  } else {
    if (!yColumnByteStride) {
      MatrixTimesMatrix<RCAT, RKIND, XT, YT, true, false>(
          product, rows, cols, x, y, n, *xColumnByteStride);
    } else {
      MatrixTimesMatrix<RCAT, RKIND, XT, YT, true, true>(
          product, rows, cols, x, y, n, *xColumnByteStride, *yColumnByteStride);
    }
  }
}

RT_DIAG_PUSH
RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN

// Contiguous numeric matrix*vector multiplication
//   matrix(rows,n) * column vector(n) -> column vector(rows)
// Straightforward algorithm:
//   DO 1 J = 1, NROWS
//    RES(J) = 0
//    DO 1 K = 1, N
//   1 RES(J) = RES(J) + X(J,K)*Y(K)
// With loop distribution and transposition to avoid the inner
// sum reduction and to avoid non-unit strides:
//   DO 1 J = 1, NROWS
//   1 RES(J) = 0
//   DO 2 K = 1, N
//    DO 2 J = 1, NROWS
//   2 RES(J) = RES(J) + X(J,K)*Y(K)
template <TypeCategory RCAT, int RKIND, typename XT, typename YT,
    bool X_HAS_STRIDED_COLUMNS>
inline RT_API_ATTRS void MatrixTimesVector(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue n, const XT *RESTRICT x, const YT *RESTRICT y,
    std::size_t xColumnByteStride = 0) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, rows * sizeof *product);
  [[maybe_unused]] const XT *RESTRICT xp0{x};
  for (SubscriptValue k{0}; k < n; ++k) {
    ResultType *RESTRICT p{product};
    auto yv{static_cast<ResultType>(*y++)};
    for (SubscriptValue j{0}; j < rows; ++j) {
      *p++ += static_cast<ResultType>(*x++) * yv;
    }
    if constexpr (X_HAS_STRIDED_COLUMNS) {
      xp0 = reinterpret_cast<const XT *>(
          reinterpret_cast<const char *>(xp0) + xColumnByteStride);
      x = xp0;
    }
  }
}

RT_DIAG_POP

template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline RT_API_ATTRS void MatrixTimesVectorHelper(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue rows,
    SubscriptValue n, const XT *RESTRICT x, const YT *RESTRICT y,
    std::optional<std::size_t> xColumnByteStride) {
  if (!xColumnByteStride) {
    MatrixTimesVector<RCAT, RKIND, XT, YT, false>(product, rows, n, x, y);
  } else {
    MatrixTimesVector<RCAT, RKIND, XT, YT, true>(
        product, rows, n, x, y, *xColumnByteStride);
  }
}

RT_DIAG_PUSH
RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN

// Contiguous numeric vector*matrix multiplication
//   row vector(n) * matrix(n,cols) -> row vector(cols)
// Straightforward algorithm:
//   DO 1 J = 1, NCOLS
//    RES(J) = 0
//    DO 1 K = 1, N
//   1 RES(J) = RES(J) + X(K)*Y(K,J)
// With loop distribution and transposition to avoid the inner
// sum reduction and one non-unit stride (the other remains):
//   DO 1 J = 1, NCOLS
//   1 RES(J) = 0
//   DO 2 K = 1, N
//    DO 2 J = 1, NCOLS
//   2 RES(J) = RES(J) + X(K)*Y(K,J)
template <TypeCategory RCAT, int RKIND, typename XT, typename YT,
    bool Y_HAS_STRIDED_COLUMNS>
inline RT_API_ATTRS void VectorTimesMatrix(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue n,
    SubscriptValue cols, const XT *RESTRICT x, const YT *RESTRICT y,
    std::size_t yColumnByteStride = 0) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, cols * sizeof *product);
  for (SubscriptValue k{0}; k < n; ++k) {
    ResultType *RESTRICT p{product};
    auto xv{static_cast<ResultType>(*x++)};
    const YT *RESTRICT yp{&y[k]};
    for (SubscriptValue j{0}; j < cols; ++j) {
      *p++ += xv * static_cast<ResultType>(*yp);
      if constexpr (!Y_HAS_STRIDED_COLUMNS) {
        yp += n;
      } else {
        yp = reinterpret_cast<const YT *>(
            reinterpret_cast<const char *>(yp) + yColumnByteStride);
      }
    }
  }
}

RT_DIAG_POP

template <TypeCategory RCAT, int RKIND, typename XT, typename YT,
    bool SPARSE_COLUMNS = false>
inline RT_API_ATTRS void VectorTimesMatrixHelper(
    CppTypeFor<RCAT, RKIND> *RESTRICT product, SubscriptValue n,
    SubscriptValue cols, const XT *RESTRICT x, const YT *RESTRICT y,
    std::optional<std::size_t> yColumnByteStride) {
  if (!yColumnByteStride) {
    VectorTimesMatrix<RCAT, RKIND, XT, YT, false>(product, n, cols, x, y);
  } else {
    VectorTimesMatrix<RCAT, RKIND, XT, YT, true>(
        product, n, cols, x, y, *yColumnByteStride);
  }
}

RT_DIAG_PUSH
RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN

// Implements an instance of MATMUL for given argument types.
template <bool IS_ALLOCATING, TypeCategory RCAT, int RKIND, typename XT,
    typename YT>
static inline RT_API_ATTRS void DoMatmul(
    std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor> &result,
    const Descriptor &x, const Descriptor &y, Terminator &terminator) {
  int xRank{x.rank()};
  int yRank{y.rank()};
  int resRank{xRank + yRank - 2};
  if (xRank * yRank != 2 * resRank) {
    terminator.Crash("MATMUL: bad argument ranks (%d * %d)", xRank, yRank);
  }
  SubscriptValue extent[2]{
      xRank == 2 ? x.GetDimension(0).Extent() : y.GetDimension(1).Extent(),
      resRank == 2 ? y.GetDimension(1).Extent() : 0};
  if constexpr (IS_ALLOCATING) {
    result.Establish(
        RCAT, RKIND, nullptr, resRank, extent, CFI_attribute_allocatable);
    for (int j{0}; j < resRank; ++j) {
      result.GetDimension(j).SetBounds(1, extent[j]);
    }
    if (int stat{result.Allocate()}) {
      terminator.Crash(
          "MATMUL: could not allocate memory for result; STAT=%d", stat);
    }
  } else {
    RUNTIME_CHECK(terminator, resRank == result.rank());
    RUNTIME_CHECK(
        terminator, result.ElementBytes() == static_cast<std::size_t>(RKIND));
    RUNTIME_CHECK(terminator, result.GetDimension(0).Extent() == extent[0]);
    RUNTIME_CHECK(terminator,
        resRank == 1 || result.GetDimension(1).Extent() == extent[1]);
  }
  SubscriptValue n{x.GetDimension(xRank - 1).Extent()};
  if (n != y.GetDimension(0).Extent()) {
    terminator.Crash("MATMUL: unacceptable operand shapes (%jdx%jd, %jdx%jd)",
        static_cast<std::intmax_t>(x.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(n),
        static_cast<std::intmax_t>(y.GetDimension(0).Extent()),
        static_cast<std::intmax_t>(y.GetDimension(1).Extent()));
  }
  using WriteResult =
      CppTypeFor<RCAT == TypeCategory::Logical ? TypeCategory::Integer : RCAT,
          RKIND>;
  if constexpr (RCAT != TypeCategory::Logical) {
    if (x.IsContiguous(1) && y.IsContiguous(1) &&
        (IS_ALLOCATING || result.IsContiguous())) {
      // Contiguous numeric matrices (maybe with columns
      // separated by a stride).
      std::optional<std::size_t> xColumnByteStride;
      if (!x.IsContiguous()) {
        // X's columns are strided.
        SubscriptValue xAt[2]{};
        x.GetLowerBounds(xAt);
        xAt[1]++;
        xColumnByteStride = x.SubscriptsToByteOffset(xAt);
      }
      std::optional<std::size_t> yColumnByteStride;
      if (!y.IsContiguous()) {
        // Y's columns are strided.
        SubscriptValue yAt[2]{};
        y.GetLowerBounds(yAt);
        yAt[1]++;
        yColumnByteStride = y.SubscriptsToByteOffset(yAt);
      }
      // Note that BLAS GEMM can be used for the strided
      // columns by setting proper leading dimension size.
      // This implies that the column stride is divisible
      // by the element size, which is usually true.
      if (resRank == 2) { // M*M -> M
        if (std::is_same_v<XT, YT>) {
          if constexpr (std::is_same_v<XT, float>) {
            // TODO: call BLAS-3 SGEMM
            // TODO: try using CUTLASS for device.
          } else if constexpr (std::is_same_v<XT, double>) {
            // TODO: call BLAS-3 DGEMM
          } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
            // TODO: call BLAS-3 CGEMM
          } else if constexpr (std::is_same_v<XT, std::complex<double>>) {
            // TODO: call BLAS-3 ZGEMM
          }
        }
        MatrixTimesMatrixHelper<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), extent[0], extent[1],
            x.OffsetElement<XT>(), y.OffsetElement<YT>(), n, xColumnByteStride,
            yColumnByteStride);
        return;
      } else if (xRank == 2) { // M*V -> V
        if (std::is_same_v<XT, YT>) {
          if constexpr (std::is_same_v<XT, float>) {
            // TODO: call BLAS-2 SGEMV(x,y)
          } else if constexpr (std::is_same_v<XT, double>) {
            // TODO: call BLAS-2 DGEMV(x,y)
          } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
            // TODO: call BLAS-2 CGEMV(x,y)
          } else if constexpr (std::is_same_v<XT, std::complex<double>>) {
            // TODO: call BLAS-2 ZGEMV(x,y)
          }
        }
        MatrixTimesVectorHelper<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), extent[0], n,
            x.OffsetElement<XT>(), y.OffsetElement<YT>(), xColumnByteStride);
        return;
      } else { // V*M -> V
        if (std::is_same_v<XT, YT>) {
          if constexpr (std::is_same_v<XT, float>) {
            // TODO: call BLAS-2 SGEMV(y,x)
          } else if constexpr (std::is_same_v<XT, double>) {
            // TODO: call BLAS-2 DGEMV(y,x)
          } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
            // TODO: call BLAS-2 CGEMV(y,x)
          } else if constexpr (std::is_same_v<XT, std::complex<double>>) {
            // TODO: call BLAS-2 ZGEMV(y,x)
          }
        }
        VectorTimesMatrixHelper<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), n, extent[0],
            x.OffsetElement<XT>(), y.OffsetElement<YT>(), yColumnByteStride);
        return;
      }
    }
  }
  // General algorithms for LOGICAL and noncontiguity
  SubscriptValue xAt[2], yAt[2], resAt[2];
  x.GetLowerBounds(xAt);
  y.GetLowerBounds(yAt);
  result.GetLowerBounds(resAt);
  if (resRank == 2) { // M*M -> M
    SubscriptValue x1{xAt[1]}, y0{yAt[0]}, y1{yAt[1]}, res1{resAt[1]};
    for (SubscriptValue i{0}; i < extent[0]; ++i) {
      for (SubscriptValue j{0}; j < extent[1]; ++j) {
        Accumulator<RCAT, RKIND, XT, YT> accumulator{x, y};
        yAt[1] = y1 + j;
        for (SubscriptValue k{0}; k < n; ++k) {
          xAt[1] = x1 + k;
          yAt[0] = y0 + k;
          accumulator.Accumulate(xAt, yAt);
        }
        resAt[1] = res1 + j;
        *result.template Element<WriteResult>(resAt) = accumulator.GetResult();
      }
      ++resAt[0];
      ++xAt[0];
    }
  } else if (xRank == 2) { // M*V -> V
    SubscriptValue x1{xAt[1]}, y0{yAt[0]};
    for (SubscriptValue j{0}; j < extent[0]; ++j) {
      Accumulator<RCAT, RKIND, XT, YT> accumulator{x, y};
      for (SubscriptValue k{0}; k < n; ++k) {
        xAt[1] = x1 + k;
        yAt[0] = y0 + k;
        accumulator.Accumulate(xAt, yAt);
      }
      *result.template Element<WriteResult>(resAt) = accumulator.GetResult();
      ++resAt[0];
      ++xAt[0];
    }
  } else { // V*M -> V
    SubscriptValue x0{xAt[0]}, y0{yAt[0]};
    for (SubscriptValue j{0}; j < extent[0]; ++j) {
      Accumulator<RCAT, RKIND, XT, YT> accumulator{x, y};
      for (SubscriptValue k{0}; k < n; ++k) {
        xAt[0] = x0 + k;
        yAt[0] = y0 + k;
        accumulator.Accumulate(xAt, yAt);
      }
      *result.template Element<WriteResult>(resAt) = accumulator.GetResult();
      ++resAt[0];
      ++yAt[1];
    }
  }
}

RT_DIAG_POP

// Maps the dynamic type information from the arguments' descriptors
// to the right instantiation of DoMatmul() for valid combinations of
// types.
template <bool IS_ALLOCATING> struct Matmul {
  using ResultDescriptor =
      std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor>;
  template <TypeCategory XCAT, int XKIND> struct MM1 {
    template <TypeCategory YCAT, int YKIND> struct MM2 {
      RT_API_ATTRS void operator()(ResultDescriptor &result,
          const Descriptor &x, const Descriptor &y,
          Terminator &terminator) const {
        if constexpr (constexpr auto resultType{
                          GetResultType(XCAT, XKIND, YCAT, YKIND)}) {
          if constexpr (common::IsNumericTypeCategory(resultType->first) ||
              resultType->first == TypeCategory::Logical) {
            return DoMatmul<IS_ALLOCATING, resultType->first,
                resultType->second, CppTypeFor<XCAT, XKIND>,
                CppTypeFor<YCAT, YKIND>>(result, x, y, terminator);
          }
        }
        terminator.Crash("MATMUL: bad operand types (%d(%d), %d(%d))",
            static_cast<int>(XCAT), XKIND, static_cast<int>(YCAT), YKIND);
      }
    };
    RT_API_ATTRS void operator()(ResultDescriptor &result, const Descriptor &x,
        const Descriptor &y, Terminator &terminator, TypeCategory yCat,
        int yKind) const {
      ApplyType<MM2, void>(yCat, yKind, terminator, result, x, y, terminator);
    }
  };
  RT_API_ATTRS void operator()(ResultDescriptor &result, const Descriptor &x,
      const Descriptor &y, const char *sourceFile, int line) const {
    Terminator terminator{sourceFile, line};
    auto xCatKind{x.type().GetCategoryAndKind()};
    auto yCatKind{y.type().GetCategoryAndKind()};
    RUNTIME_CHECK(terminator, xCatKind.has_value() && yCatKind.has_value());
    ApplyType<MM1, void>(xCatKind->first, xCatKind->second, terminator, result,
        x, y, terminator, yCatKind->first, yCatKind->second);
  }
};

extern "C" {
void RTDEF(Matmul)(Descriptor &result, const Descriptor &x, const Descriptor &y,
    const char *sourceFile, int line) {
  Matmul<true>{}(result, x, y, sourceFile, line);
}
void RTDEF(MatmulDirect)(const Descriptor &result, const Descriptor &x,
    const Descriptor &y, const char *sourceFile, int line) {
  Matmul<false>{}(result, x, y, sourceFile, line);
}
} // extern "C"
} // namespace Fortran::runtime
