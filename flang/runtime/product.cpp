//===-- runtime/product.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements PRODUCT for all required operand types and shapes.

#include "reduction-templates.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/reduction.h"
#include <cfloat>
#include <cinttypes>
#include <complex>

namespace Fortran::runtime {
template <typename INTERMEDIATE> class NonComplexProductAccumulator {
public:
  explicit RT_API_ATTRS NonComplexProductAccumulator(const Descriptor &array)
      : array_{array} {}
  RT_API_ATTRS void Reinitialize() { product_ = 1; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(product_);
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    product_ *= *array_.Element<A>(at);
    return product_ != 0;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE product_{1};
};

// Suppress the warnings about calling __host__-only std::complex operators,
// defined in C++ STD header files, from __device__ code.
RT_DIAG_PUSH
RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN

template <typename PART> class ComplexProductAccumulator {
public:
  explicit RT_API_ATTRS ComplexProductAccumulator(const Descriptor &array)
      : array_{array} {}
  RT_API_ATTRS void Reinitialize() { product_ = std::complex<PART>{1, 0}; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    using ResultPart = typename A::value_type;
    *p = {static_cast<ResultPart>(product_.real()),
        static_cast<ResultPart>(product_.imag())};
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    product_ *= *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  std::complex<PART> product_{1, 0};
};

RT_DIAG_POP

extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Integer, 1> RTDEF(ProductInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(ProductInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(ProductInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(ProductInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x},
      "PRODUCT");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(ProductInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "PRODUCT");
}
#endif

// TODO: real/complex(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(ProductReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 4>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(ProductReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 8>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTDEF(ProductReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 10>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 10>>{x},
      "PRODUCT");
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(ProductReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 16>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 16>>{x},
      "PRODUCT");
}
#endif

void RTDEF(CppProductComplex4)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 4>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 4>>{x},
      "PRODUCT");
}
void RTDEF(CppProductComplex8)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 8>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
#if LDBL_MANT_DIG == 64
void RTDEF(CppProductComplex10)(CppTypeFor<TypeCategory::Complex, 10> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 10>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 10>>{x},
      "PRODUCT");
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
void RTDEF(CppProductComplex16)(CppTypeFor<TypeCategory::Complex, 16> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 16>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 16>>{x},
      "PRODUCT");
}
#endif

void RTDEF(ProductDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  TypedPartialNumericReduction<NonComplexProductAccumulator,
      NonComplexProductAccumulator, ComplexProductAccumulator,
      /*MIN_REAL_KIND=*/4>(result, x, dim, source, line, mask, "PRODUCT");
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
