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

// General accumulator for any type and stride; this is not used for
// contiguous numeric cases.
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
class Accumulator {
public:
  using Result = AccumulationType<RCAT, RKIND>;
  Accumulator(const Descriptor &x, const Descriptor &y) : x_{x}, y_{y} {}
  void Accumulate(const SubscriptValue xAt[], const SubscriptValue yAt[]) {
    if constexpr (RCAT == TypeCategory::Logical) {
      sum_ = sum_ ||
          (IsLogicalElementTrue(x_, xAt) && IsLogicalElementTrue(y_, yAt));
    } else {
      sum_ += static_cast<Result>(*x_.Element<XT>(xAt)) *
          static_cast<Result>(*y_.Element<YT>(yAt));
    }
  }
  Result GetResult() const { return sum_; }

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
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline void MatrixTimesMatrix(CppTypeFor<RCAT, RKIND> *RESTRICT product,
    SubscriptValue rows, SubscriptValue cols, const XT *RESTRICT x,
    const YT *RESTRICT y, SubscriptValue n) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, rows * cols * sizeof *product);
  const XT *RESTRICT xp0{x};
  for (SubscriptValue k{0}; k < n; ++k) {
    ResultType *RESTRICT p{product};
    for (SubscriptValue j{0}; j < cols; ++j) {
      const XT *RESTRICT xp{xp0};
      auto yv{static_cast<ResultType>(y[k + j * n])};
      for (SubscriptValue i{0}; i < rows; ++i) {
        *p++ += static_cast<ResultType>(*xp++) * yv;
      }
    }
    xp0 += rows;
  }
}

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
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline void MatrixTimesVector(CppTypeFor<RCAT, RKIND> *RESTRICT product,
    SubscriptValue rows, SubscriptValue n, const XT *RESTRICT x,
    const YT *RESTRICT y) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, rows * sizeof *product);
  for (SubscriptValue k{0}; k < n; ++k) {
    ResultType *RESTRICT p{product};
    auto yv{static_cast<ResultType>(*y++)};
    for (SubscriptValue j{0}; j < rows; ++j) {
      *p++ += static_cast<ResultType>(*x++) * yv;
    }
  }
}

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
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
inline void VectorTimesMatrix(CppTypeFor<RCAT, RKIND> *RESTRICT product,
    SubscriptValue n, SubscriptValue cols, const XT *RESTRICT x,
    const YT *RESTRICT y) {
  using ResultType = CppTypeFor<RCAT, RKIND>;
  std::memset(product, 0, cols * sizeof *product);
  for (SubscriptValue k{0}; k < n; ++k) {
    ResultType *RESTRICT p{product};
    auto xv{static_cast<ResultType>(*x++)};
    const YT *RESTRICT yp{&y[k]};
    for (SubscriptValue j{0}; j < cols; ++j) {
      *p++ += xv * static_cast<ResultType>(*yp);
      yp += n;
    }
  }
}

// Implements an instance of MATMUL for given argument types.
template <bool IS_ALLOCATING, TypeCategory RCAT, int RKIND, typename XT,
    typename YT>
static inline void DoMatmul(
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
    terminator.Crash("MATMUL: arrays do not conform (%jd != %jd)",
        static_cast<std::intmax_t>(n),
        static_cast<std::intmax_t>(y.GetDimension(0).Extent()));
  }
  using WriteResult =
      CppTypeFor<RCAT == TypeCategory::Logical ? TypeCategory::Integer : RCAT,
          RKIND>;
  if constexpr (RCAT != TypeCategory::Logical) {
    if (x.IsContiguous() && y.IsContiguous() &&
        (IS_ALLOCATING || result.IsContiguous())) {
      // Contiguous numeric matrices
      if (resRank == 2) { // M*M -> M
        if (std::is_same_v<XT, YT>) {
          if constexpr (std::is_same_v<XT, float>) {
            // TODO: call BLAS-3 SGEMM
          } else if constexpr (std::is_same_v<XT, double>) {
            // TODO: call BLAS-3 DGEMM
          } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
            // TODO: call BLAS-3 CGEMM
          } else if constexpr (std::is_same_v<XT, std::complex<double>>) {
            // TODO: call BLAS-3 ZGEMM
          }
        }
        MatrixTimesMatrix<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), extent[0], extent[1],
            x.OffsetElement<XT>(), y.OffsetElement<YT>(), n);
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
        MatrixTimesVector<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), extent[0], n,
            x.OffsetElement<XT>(), y.OffsetElement<YT>());
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
        VectorTimesMatrix<RCAT, RKIND, XT, YT>(
            result.template OffsetElement<WriteResult>(), n, extent[0],
            x.OffsetElement<XT>(), y.OffsetElement<YT>());
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

// Maps the dynamic type information from the arguments' descriptors
// to the right instantiation of DoMatmul() for valid combinations of
// types.
template <bool IS_ALLOCATING> struct Matmul {
  using ResultDescriptor =
      std::conditional_t<IS_ALLOCATING, Descriptor, const Descriptor>;
  template <TypeCategory XCAT, int XKIND> struct MM1 {
    template <TypeCategory YCAT, int YKIND> struct MM2 {
      void operator()(ResultDescriptor &result, const Descriptor &x,
          const Descriptor &y, Terminator &terminator) const {
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

extern "C" {
void RTNAME(Matmul)(Descriptor &result, const Descriptor &x,
    const Descriptor &y, const char *sourceFile, int line) {
  Matmul<true>{}(result, x, y, sourceFile, line);
}
void RTNAME(MatmulDirect)(const Descriptor &result, const Descriptor &x,
    const Descriptor &y, const char *sourceFile, int line) {
  Matmul<false>{}(result, x, y, sourceFile, line);
}
} // extern "C"
} // namespace Fortran::runtime
