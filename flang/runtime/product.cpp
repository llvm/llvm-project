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
  explicit NonComplexProductAccumulator(const Descriptor &array)
      : array_{array} {}
  void Reinitialize() { product_ = 1; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(product_);
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    product_ *= *array_.Element<A>(at);
    return product_ != 0;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE product_{1};
};

template <typename PART> class ComplexProductAccumulator {
public:
  explicit ComplexProductAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { product_ = std::complex<PART>{1, 0}; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    using ResultPart = typename A::value_type;
    *p = {static_cast<ResultPart>(product_.real()),
        static_cast<ResultPart>(product_.imag())};
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    product_ *= *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  std::complex<PART> product_{1, 0};
};

extern "C" {
CppTypeFor<TypeCategory::Integer, 1> RTNAME(ProductInteger1)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(ProductInteger2)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(ProductInteger4)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(ProductInteger8)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x},
      "PRODUCT");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(ProductInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "PRODUCT");
}
#endif

// TODO: real/complex(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(ProductReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 4>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(ProductReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 8>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
#if LDBL_MANT_DIG == 64
CppTypeFor<TypeCategory::Real, 10> RTNAME(ProductReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 10>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 10>>{x},
      "PRODUCT");
}
#elif LDBL_MANT_DIG == 113
CppTypeFor<TypeCategory::Real, 16> RTNAME(ProductReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 16>(x, source, line, dim, mask,
      NonComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 16>>{x},
      "PRODUCT");
}
#endif

void RTNAME(CppProductComplex4)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 4>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
void RTNAME(CppProductComplex8)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 8>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 8>>{x},
      "PRODUCT");
}
#if LDBL_MANT_DIG == 64
void RTNAME(CppProductComplex10)(CppTypeFor<TypeCategory::Complex, 10> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 10>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 10>>{x},
      "PRODUCT");
}
#elif LDBL_MANT_DIG == 113
void RTNAME(CppProductComplex16)(CppTypeFor<TypeCategory::Complex, 16> &result,
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  result = GetTotalReduction<TypeCategory::Complex, 16>(x, source, line, dim,
      mask, ComplexProductAccumulator<CppTypeFor<TypeCategory::Real, 16>>{x},
      "PRODUCT");
}
#endif

void RTNAME(ProductDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  TypedPartialNumericReduction<NonComplexProductAccumulator,
      NonComplexProductAccumulator, ComplexProductAccumulator>(
      result, x, dim, source, line, mask, "PRODUCT");
}
} // extern "C"
} // namespace Fortran::runtime
