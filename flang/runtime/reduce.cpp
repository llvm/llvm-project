//===-- runtime/reduce.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REDUCE() implementation

#include "flang/Runtime/reduce.h"
#include "reduction-templates.h"
#include "terminator.h"
#include "tools.h"
#include "flang/Runtime/descriptor.h"

namespace Fortran::runtime {

template <typename T> class ReduceAccumulator {
public:
  RT_API_ATTRS ReduceAccumulator(const Descriptor &array,
      ReductionOperation<T> operation, const T *identity,
      Terminator &terminator)
      : array_{array}, operation_{operation}, identity_{identity},
        terminator_{terminator} {}
  RT_API_ATTRS void Reinitialize() { result_.reset(); }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    const auto *operand{array_.Element<A>(at)};
    if (result_) {
      result_ = operation_(&*result_, operand);
    } else {
      result_ = *operand;
    }
    return true;
  }
  template <typename A>
  RT_API_ATTRS void GetResult(A *to, int /*zeroBasedDim*/ = -1) {
    if (result_) {
      *to = *result_;
    } else if (identity_) {
      *to = *identity_;
    } else {
      terminator_.Crash("REDUCE() without IDENTITY= has no result");
    }
  }

private:
  const Descriptor &array_;
  common::optional<T> result_;
  ReductionOperation<T> operation_;
  const T *identity_{nullptr};
  Terminator &terminator_;
};

template <typename T, typename OP, bool hasLength>
class BufferedReduceAccumulator {
public:
  RT_API_ATTRS BufferedReduceAccumulator(const Descriptor &array, OP operation,
      const T *identity, Terminator &terminator)
      : array_{array}, operation_{operation}, identity_{identity},
        terminator_{terminator} {}
  RT_API_ATTRS void Reinitialize() { activeTemp_ = -1; }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    const auto *operand{array_.Element<A>(at)};
    if (activeTemp_ >= 0) {
      if constexpr (hasLength) {
        operation_(&*temp_[1 - activeTemp_], length_, &*temp_[activeTemp_],
            operand, length_, length_);
      } else {
        operation_(&*temp_[1 - activeTemp_], &*temp_[activeTemp_], operand);
      }
      activeTemp_ = 1 - activeTemp_;
    } else {
      activeTemp_ = 0;
      std::memcpy(&*temp_[activeTemp_], operand, elementBytes_);
    }
    return true;
  }
  template <typename A>
  RT_API_ATTRS void GetResult(A *to, int /*zeroBasedDim*/ = -1) {
    if (activeTemp_ >= 0) {
      std::memcpy(to, &*temp_[activeTemp_], elementBytes_);
    } else if (identity_) {
      std::memcpy(to, identity_, elementBytes_);
    } else {
      terminator_.Crash("REDUCE() without IDENTITY= has no result");
    }
  }

private:
  const Descriptor &array_;
  OP operation_;
  const T *identity_{nullptr};
  Terminator &terminator_;
  std::size_t elementBytes_{array_.ElementBytes()};
  OwningPtr<T> temp_[2]{SizedNew<T>{terminator_}(elementBytes_),
      SizedNew<T>{terminator_}(elementBytes_)};
  int activeTemp_{-1};
  std::size_t length_{elementBytes_ / sizeof(T)};
};

extern "C" {
RT_EXT_API_GROUP_BEGIN

std::int8_t RTDEF(ReduceInteger1)(const Descriptor &array,
    ReductionOperation<std::int8_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 1>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int8_t>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger1Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int8_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int8_t>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 1>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::int16_t RTDEF(ReduceInteger2)(const Descriptor &array,
    ReductionOperation<std::int16_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 2>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int16_t>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger2Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int16_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int16_t>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 2>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::int32_t RTDEF(ReduceInteger4)(const Descriptor &array,
    ReductionOperation<std::int32_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int32_t>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int32_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int32_t>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::int64_t RTDEF(ReduceInteger8)(const Descriptor &array,
    ReductionOperation<std::int64_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int64_t>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int64_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int64_t>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#ifdef __SIZEOF_INT128__
common::int128_t RTDEF(ReduceInteger16)(const Descriptor &array,
    ReductionOperation<common::int128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const common::int128_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<common::int128_t>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger16Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<common::int128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const common::int128_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<common::int128_t>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif

// TODO: real/complex(2 & 3)
float RTDEF(ReduceReal4)(const Descriptor &array,
    ReductionOperation<float> operation, const char *source, int line, int dim,
    const Descriptor *mask, const float *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 4>(array, source, line, dim,
      mask, ReduceAccumulator<float>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<float> operation, const char *source, int line, int dim,
    const Descriptor *mask, const float *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<float>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
double RTDEF(ReduceReal8)(const Descriptor &array,
    ReductionOperation<double> operation, const char *source, int line, int dim,
    const Descriptor *mask, const double *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 8>(array, source, line, dim,
      mask, ReduceAccumulator<double>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<double> operation, const char *source, int line, int dim,
    const Descriptor *mask, const double *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<double>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#if LDBL_MANT_DIG == 64
long double RTDEF(ReduceReal10)(const Descriptor &array,
    ReductionOperation<long double> operation, const char *source, int line,
    int dim, const Descriptor *mask, const long double *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 10>(array, source, line, dim,
      mask,
      ReduceAccumulator<long double>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal10Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<long double> operation, const char *source, int line,
    int dim, const Descriptor *mask, const long double *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<long double>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 10>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
CppFloat128Type RTDEF(ReduceReal16)(const Descriptor &array,
    ReductionOperation<CppFloat128Type> operation, const char *source, int line,
    int dim, const Descriptor *mask, const CppFloat128Type *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppFloat128Type>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal16Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<CppFloat128Type> operation, const char *source, int line,
    int dim, const Descriptor *mask, const CppFloat128Type *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<CppFloat128Type>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif

void RTDEF(CppReduceComplex4)(std::complex<float> &result,
    const Descriptor &array, ReductionOperation<std::complex<float>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const std::complex<float> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::complex<float>>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<float>> operation, const char *source,
    int line, int dim, const Descriptor *mask,
    const std::complex<float> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::complex<float>>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(CppReduceComplex8)(std::complex<double> &result,
    const Descriptor &array, ReductionOperation<std::complex<double>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const std::complex<double> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::complex<double>>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<double>> operation, const char *source,
    int line, int dim, const Descriptor *mask,
    const std::complex<double> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::complex<double>>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#if LDBL_MANT_DIG == 64
void RTDEF(CppReduceComplex10)(std::complex<long double> &result,
    const Descriptor &array,
    ReductionOperation<std::complex<long double>> operation, const char *source,
    int line, int dim, const Descriptor *mask,
    const std::complex<long double> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 10>(array, source, line,
      dim, mask,
      ReduceAccumulator<std::complex<long double>>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex10Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<long double>> operation, const char *source,
    int line, int dim, const Descriptor *mask,
    const std::complex<long double> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::complex<long double>>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 10>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif
#if LDBL_MANT_DIG == 113 || HAS_FLOAT128
void RTDEF(CppReduceComplex16)(std::complex<CppFloat128Type> &result,
    const Descriptor &array,
    ReductionOperation<std::complex<CppFloat128Type>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const std::complex<CppFloat128Type> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 16>(array, source, line,
      dim, mask,
      ReduceAccumulator<std::complex<CppFloat128Type>>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex16Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::complex<CppFloat128Type>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const std::complex<CppFloat128Type> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::complex<CppFloat128Type>>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif

bool RTDEF(ReduceLogical1)(const Descriptor &array,
    ReductionOperation<std::int8_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger1)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical1Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int8_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger1Dim)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
bool RTDEF(ReduceLogical2)(const Descriptor &array,
    ReductionOperation<std::int16_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger2)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical2Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int16_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger2Dim)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
bool RTDEF(ReduceLogical4)(const Descriptor &array,
    ReductionOperation<std::int32_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger4)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical4Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int32_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger4Dim)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
bool RTDEF(ReduceLogical8)(const Descriptor &array,
    ReductionOperation<std::int64_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger8)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical8Dim)(Descriptor &result, const Descriptor &array,
    ReductionOperation<std::int64_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger8Dim)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}

void RTDEF(ReduceChar1)(char *result, const Descriptor &array,
    ReductionCharOperation<char> operation, const char *source, int line,
    int dim, const Descriptor *mask, const char *identity, bool ordered) {
  Terminator terminator{source, line};
  BufferedReduceAccumulator<char, ReductionCharOperation<char>,
      /*hasLength=*/true>
      accumulator{array, operation, identity, terminator};
  DoTotalReduction<char>(array, dim, mask, accumulator, "REDUCE", terminator);
  accumulator.GetResult(result);
}
void RTDEF(ReduceCharacter1Dim)(Descriptor &result, const Descriptor &array,
    ReductionCharOperation<char> operation, const char *source, int line,
    int dim, const Descriptor *mask, const char *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = BufferedReduceAccumulator<char,
      ReductionCharOperation<char>, /*hasLength=*/true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Character, 1>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceChar2)(char16_t *result, const Descriptor &array,
    ReductionCharOperation<char16_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const char16_t *identity, bool ordered) {
  Terminator terminator{source, line};
  BufferedReduceAccumulator<char16_t, ReductionCharOperation<char16_t>,
      /*hasLength=*/true>
      accumulator{array, operation, identity, terminator};
  DoTotalReduction<char16_t>(
      array, dim, mask, accumulator, "REDUCE", terminator);
  accumulator.GetResult(result);
}
void RTDEF(ReduceCharacter2Dim)(Descriptor &result, const Descriptor &array,
    ReductionCharOperation<char16_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const char16_t *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = BufferedReduceAccumulator<char16_t,
      ReductionCharOperation<char16_t>, /*hasLength=*/true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Character, 2>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceChar4)(char32_t *result, const Descriptor &array,
    ReductionCharOperation<char32_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const char32_t *identity, bool ordered) {
  Terminator terminator{source, line};
  BufferedReduceAccumulator<char32_t, ReductionCharOperation<char32_t>,
      /*hasLength=*/true>
      accumulator{array, operation, identity, terminator};
  DoTotalReduction<char32_t>(
      array, dim, mask, accumulator, "REDUCE", terminator);
  accumulator.GetResult(result);
}
void RTDEF(ReduceCharacter4Dim)(Descriptor &result, const Descriptor &array,
    ReductionCharOperation<char32_t> operation, const char *source, int line,
    int dim, const Descriptor *mask, const char32_t *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = BufferedReduceAccumulator<char32_t,
      ReductionCharOperation<char32_t>, /*hasLength=*/true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Character, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}

void RTDEF(ReduceDerivedType)(char *result, const Descriptor &array,
    ReductionDerivedTypeOperation operation, const char *source, int line,
    int dim, const Descriptor *mask, const char *identity, bool ordered) {
  Terminator terminator{source, line};
  BufferedReduceAccumulator<char, ReductionDerivedTypeOperation,
      /*hasLength=*/false>
      accumulator{array, operation, identity, terminator};
  DoTotalReduction<char>(array, dim, mask, accumulator, "REDUCE", terminator);
  accumulator.GetResult(result);
}
void RTDEF(ReduceDerivedTypeDim)(Descriptor &result, const Descriptor &array,
    ReductionDerivedTypeOperation operation, const char *source, int line,
    int dim, const Descriptor *mask, const char *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = BufferedReduceAccumulator<char,
      ReductionDerivedTypeOperation, /*hasLength=*/false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Derived, 0>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
