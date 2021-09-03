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
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
class Accumulator {
public:
  // Accumulate floating-point results in (at least) double precision
  using Result = CppTypeFor<RCAT,
      RCAT == TypeCategory::Real || RCAT == TypeCategory::Complex
          ? std::max(RKIND, static_cast<int>(sizeof(double)))
          : RKIND>;
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
    RUNTIME_CHECK(terminator, result.type() == (TypeCode{RCAT, RKIND}));
    RUNTIME_CHECK(terminator, result.GetDimension(0).Extent() == extent[0]);
    RUNTIME_CHECK(terminator,
        resRank == 1 || result.GetDimension(1).Extent() == extent[1]);
  }
  using WriteResult =
      CppTypeFor<RCAT == TypeCategory::Logical ? TypeCategory::Integer : RCAT,
          RKIND>;
  SubscriptValue n{x.GetDimension(xRank - 1).Extent()};
  if (n != y.GetDimension(0).Extent()) {
    terminator.Crash("MATMUL: arrays do not conform (%jd != %jd)",
        static_cast<std::intmax_t>(n),
        static_cast<std::intmax_t>(y.GetDimension(0).Extent()));
  }
  SubscriptValue xAt[2], yAt[2], resAt[2];
  x.GetLowerBounds(xAt);
  y.GetLowerBounds(yAt);
  result.GetLowerBounds(resAt);
  if (resRank == 2) { // M*M -> M
    if constexpr (std::is_same_v<XT, YT>) {
      if constexpr (std::is_same_v<XT, float>) {
        // TODO: call BLAS-3 SGEMM
      } else if constexpr (std::is_same_v<XT, double>) {
        // TODO: call BLAS-3 DGEMM
      } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
        // TODO: call BLAS-3 CGEMM
      } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
        // TODO: call BLAS-3 ZGEMM
      }
    }
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
  } else {
    if constexpr (std::is_same_v<XT, YT>) {
      if constexpr (std::is_same_v<XT, float>) {
        // TODO: call BLAS-2 SGEMV
      } else if constexpr (std::is_same_v<XT, double>) {
        // TODO: call BLAS-2 DGEMV
      } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
        // TODO: call BLAS-2 CGEMV
      } else if constexpr (std::is_same_v<XT, std::complex<float>>) {
        // TODO: call BLAS-2 ZGEMV
      }
    }
    if (xRank == 2) { // M*V -> V
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
