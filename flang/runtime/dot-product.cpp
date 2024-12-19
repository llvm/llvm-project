//===-- runtime/dot-product.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "float.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/descriptor.h"
#include "flang/Runtime/reduction.h"
#include <cfloat>
#include <cinttypes>

namespace Fortran::runtime {

// Beware: DOT_PRODUCT of COMPLEX data uses the complex conjugate of the first
// argument; MATMUL does not.

// General accumulator for any type and stride; this is not used for
// contiguous numeric vectors.
template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
class Accumulator {
public:
  using Result = AccumulationType<RCAT, RKIND>;
  RT_API_ATTRS Accumulator(const Descriptor &x, const Descriptor &y)
      : x_{x}, y_{y} {}
  RT_API_ATTRS void AccumulateIndexed(SubscriptValue xAt, SubscriptValue yAt) {
    if constexpr (RCAT == TypeCategory::Logical) {
      sum_ = sum_ ||
          (IsLogicalElementTrue(x_, &xAt) && IsLogicalElementTrue(y_, &yAt));
    } else {
      const XT &xElement{*x_.Element<XT>(&xAt)};
      const YT &yElement{*y_.Element<YT>(&yAt)};
      if constexpr (RCAT == TypeCategory::Complex) {
        sum_ += rtcmplx::conj(static_cast<Result>(xElement)) *
            static_cast<Result>(yElement);
      } else {
        sum_ += static_cast<Result>(xElement) * static_cast<Result>(yElement);
      }
    }
  }
  RT_API_ATTRS Result GetResult() const { return sum_; }

private:
  const Descriptor &x_, &y_;
  Result sum_{};
};

template <TypeCategory RCAT, int RKIND, typename XT, typename YT>
static inline RT_API_ATTRS CppTypeFor<RCAT, RKIND> DoDotProduct(
    const Descriptor &x, const Descriptor &y, Terminator &terminator) {
  using Result = CppTypeFor<RCAT, RKIND>;
  RUNTIME_CHECK(terminator, x.rank() == 1 && y.rank() == 1);
  SubscriptValue n{x.GetDimension(0).Extent()};
  if (SubscriptValue yN{y.GetDimension(0).Extent()}; yN != n) {
    terminator.Crash(
        "DOT_PRODUCT: SIZE(VECTOR_A) is %jd but SIZE(VECTOR_B) is %jd",
        static_cast<std::intmax_t>(n), static_cast<std::intmax_t>(yN));
  }
  if constexpr (RCAT != TypeCategory::Logical) {
    if (x.GetDimension(0).ByteStride() == sizeof(XT) &&
        y.GetDimension(0).ByteStride() == sizeof(YT)) {
      // Contiguous numeric vectors
      if constexpr (std::is_same_v<XT, YT>) {
        // Contiguous homogeneous numeric vectors
        if constexpr (std::is_same_v<XT, float>) {
          // TODO: call BLAS-1 SDOT or SDSDOT
        } else if constexpr (std::is_same_v<XT, double>) {
          // TODO: call BLAS-1 DDOT
        } else if constexpr (std::is_same_v<XT, rtcmplx::complex<float>>) {
          // TODO: call BLAS-1 CDOTC
        } else if constexpr (std::is_same_v<XT, rtcmplx::complex<double>>) {
          // TODO: call BLAS-1 ZDOTC
        }
      }
      XT *xp{x.OffsetElement<XT>(0)};
      YT *yp{y.OffsetElement<YT>(0)};
      using AccumType = AccumulationType<RCAT, RKIND>;
      AccumType accum{};
      if constexpr (RCAT == TypeCategory::Complex) {
        for (SubscriptValue j{0}; j < n; ++j) {
          // conj() may instantiate its argument twice,
          // so xp has to be incremented separately.
          // This is a workaround for an alleged bug in clang,
          // that shows up as:
          //   warning: multiple unsequenced modifications to 'xp'
          accum += rtcmplx::conj(static_cast<AccumType>(*xp)) *
              static_cast<AccumType>(*yp++);
          xp++;
        }
      } else {
        for (SubscriptValue j{0}; j < n; ++j) {
          accum +=
              static_cast<AccumType>(*xp++) * static_cast<AccumType>(*yp++);
        }
      }
      return static_cast<Result>(accum);
    }
  }
  // Non-contiguous, heterogeneous, & LOGICAL cases
  SubscriptValue xAt{x.GetDimension(0).LowerBound()};
  SubscriptValue yAt{y.GetDimension(0).LowerBound()};
  Accumulator<RCAT, RKIND, XT, YT> accumulator{x, y};
  for (SubscriptValue j{0}; j < n; ++j) {
    accumulator.AccumulateIndexed(xAt++, yAt++);
  }
  return static_cast<Result>(accumulator.GetResult());
}

template <TypeCategory RCAT, int RKIND> struct DotProduct {
  using Result = CppTypeFor<RCAT, RKIND>;
  template <TypeCategory XCAT, int XKIND> struct DP1 {
    template <TypeCategory YCAT, int YKIND> struct DP2 {
      RT_API_ATTRS Result operator()(const Descriptor &x, const Descriptor &y,
          Terminator &terminator) const {
        if constexpr (constexpr auto resultType{
                          GetResultType(XCAT, XKIND, YCAT, YKIND)}) {
          if constexpr (resultType->first == RCAT &&
              (resultType->second <= RKIND || RCAT == TypeCategory::Logical)) {
            return DoDotProduct<RCAT, RKIND, CppTypeFor<XCAT, XKIND>,
                CppTypeFor<YCAT, YKIND>>(x, y, terminator);
          }
        }
        terminator.Crash(
            "DOT_PRODUCT(%d(%d)): bad operand types (%d(%d), %d(%d))",
            static_cast<int>(RCAT), RKIND, static_cast<int>(XCAT), XKIND,
            static_cast<int>(YCAT), YKIND);
      }
    };
    RT_API_ATTRS Result operator()(const Descriptor &x, const Descriptor &y,
        Terminator &terminator, TypeCategory yCat, int yKind) const {
      return ApplyType<DP2, Result>(yCat, yKind, terminator, x, y, terminator);
    }
  };
  RT_API_ATTRS Result operator()(const Descriptor &x, const Descriptor &y,
      const char *source, int line) const {
    Terminator terminator{source, line};
    if (RCAT != TypeCategory::Logical && x.type() == y.type()) {
      // No conversions needed, operands and result have same known type
      return typename DP1<RCAT, RKIND>::template DP2<RCAT, RKIND>{}(
          x, y, terminator);
    } else {
      auto xCatKind{x.type().GetCategoryAndKind()};
      auto yCatKind{y.type().GetCategoryAndKind()};
      RUNTIME_CHECK(terminator, xCatKind.has_value() && yCatKind.has_value());
      return ApplyType<DP1, Result>(xCatKind->first, xCatKind->second,
          terminator, x, y, terminator, yCatKind->first, yCatKind->second);
    }
  }
};

extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Integer, 1> RTDEF(DotProductInteger1)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 1>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(DotProductInteger2)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 2>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(DotProductInteger4)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 4>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(DotProductInteger8)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 8>{}(x, y, source, line);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(DotProductInteger16)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Integer, 16>{}(x, y, source, line);
}
#endif

CppTypeFor<TypeCategory::Unsigned, 1> RTDEF(DotProductUnsigned1)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 1>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Unsigned, 2> RTDEF(DotProductUnsigned2)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 2>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Unsigned, 4> RTDEF(DotProductUnsigned4)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 4>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Unsigned, 8> RTDEF(DotProductUnsigned8)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 8>{}(x, y, source, line);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Unsigned, 16> RTDEF(DotProductUnsigned16)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Unsigned, 16>{}(x, y, source, line);
}
#endif

// TODO: REAL/COMPLEX(2 & 3)
// Intermediate results and operations are at least 64 bits
CppTypeFor<TypeCategory::Real, 4> RTDEF(DotProductReal4)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 4>{}(x, y, source, line);
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(DotProductReal8)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 8>{}(x, y, source, line);
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(DotProductReal10)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 10>{}(x, y, source, line);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(DotProductReal16)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Real, 16>{}(x, y, source, line);
}
#endif

void RTDEF(CppDotProductComplex4)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 4>{}(x, y, source, line);
}
void RTDEF(CppDotProductComplex8)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 8>{}(x, y, source, line);
}
#if HAS_FLOAT80
void RTDEF(CppDotProductComplex10)(
    CppTypeFor<TypeCategory::Complex, 10> &result, const Descriptor &x,
    const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 10>{}(x, y, source, line);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(CppDotProductComplex16)(
    CppTypeFor<TypeCategory::Complex, 16> &result, const Descriptor &x,
    const Descriptor &y, const char *source, int line) {
  result = DotProduct<TypeCategory::Complex, 16>{}(x, y, source, line);
}
#endif

bool RTDEF(DotProductLogical)(
    const Descriptor &x, const Descriptor &y, const char *source, int line) {
  return DotProduct<TypeCategory::Logical, 1>{}(x, y, source, line);
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
