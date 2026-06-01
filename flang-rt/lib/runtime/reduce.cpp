//===-- lib/runtime/reduce.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REDUCE() implementation

#include "flang/Runtime/reduce.h"
#include "flang-rt/runtime/descriptor.h"
#include "flang-rt/runtime/reduction-templates.h"
#include "flang-rt/runtime/terminator.h"
#include "flang-rt/runtime/tools.h"

namespace Fortran::runtime {

template <typename T, bool isByValue> class ReduceAccumulator {
public:
  using Operation = std::conditional_t<isByValue, ValueReductionOperation<T>,
      ReferenceReductionOperation<T>>;
  RT_API_ATTRS ReduceAccumulator(const Descriptor &array, Operation operation,
      const T *identity, Terminator &terminator)
      : array_{array}, operation_{operation}, identity_{identity},
        terminator_{terminator} {}
  RT_API_ATTRS void Reinitialize() { result_.reset(); }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    const auto *operand{array_.Element<A>(at)};
    if (result_) {
      if constexpr (isByValue) {
        result_ = operation_(*result_, *operand);
      } else {
        result_ = operation_(&*result_, operand);
      }
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
  Operation operation_;
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
      runtime::memcpy(&*temp_[activeTemp_], operand, elementBytes_);
    }
    return true;
  }
  template <typename A>
  RT_API_ATTRS void GetResult(A *to, int /*zeroBasedDim*/ = -1) {
    if (activeTemp_ >= 0) {
      runtime::memcpy(to, &*temp_[activeTemp_], elementBytes_);
    } else if (identity_) {
      runtime::memcpy(to, identity_, elementBytes_);
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

std::int8_t RTDEF(ReduceInteger1Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 1>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int8_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::int8_t RTDEF(ReduceInteger1Value)(const Descriptor &array,
    ValueReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 1>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int8_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger1DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int8_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 1>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceInteger1DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int8_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 1>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::int16_t RTDEF(ReduceInteger2Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 2>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int16_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::int16_t RTDEF(ReduceInteger2Value)(const Descriptor &array,
    ValueReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 2>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int16_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger2DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int16_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 2>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceInteger2DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int16_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 2>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::int32_t RTDEF(ReduceInteger4Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int32_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::int32_t RTDEF(ReduceInteger4Value)(const Descriptor &array,
    ValueReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int32_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger4DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int32_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceInteger4DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int32_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::int64_t RTDEF(ReduceInteger8Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int64_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::int64_t RTDEF(ReduceInteger8Value)(const Descriptor &array,
    ValueReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::int64_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger8DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int64_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceInteger8DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::int64_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#ifdef __SIZEOF_INT128__
common::int128_t RTDEF(ReduceInteger16Ref)(const Descriptor &array,
    ReferenceReductionOperation<common::int128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const common::int128_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<common::int128_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
common::int128_t RTDEF(ReduceInteger16Value)(const Descriptor &array,
    ValueReductionOperation<common::int128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const common::int128_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Integer, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<common::int128_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceInteger16DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<common::int128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const common::int128_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<common::int128_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceInteger16DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<common::int128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const common::int128_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<common::int128_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Integer, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif

std::uint8_t RTDEF(ReduceUnsigned1Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::uint8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 1>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint8_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::uint8_t RTDEF(ReduceUnsigned1Value)(const Descriptor &array,
    ValueReductionOperation<std::uint8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 1>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint8_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceUnsigned1DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::uint8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint8_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 1>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceUnsigned1DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::uint8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint8_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint8_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 1>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::uint16_t RTDEF(ReduceUnsigned2Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::uint16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 2>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint16_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::uint16_t RTDEF(ReduceUnsigned2Value)(const Descriptor &array,
    ValueReductionOperation<std::uint16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 2>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint16_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceUnsigned2DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::uint16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint16_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 2>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceUnsigned2DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::uint16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint16_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint16_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 2>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::uint32_t RTDEF(ReduceUnsigned4Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::uint32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint32_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::uint32_t RTDEF(ReduceUnsigned4Value)(const Descriptor &array,
    ValueReductionOperation<std::uint32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint32_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceUnsigned4DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::uint32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint32_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceUnsigned4DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::uint32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint32_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint32_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
std::uint64_t RTDEF(ReduceUnsigned8Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::uint64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint64_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
std::uint64_t RTDEF(ReduceUnsigned8Value)(const Descriptor &array,
    ValueReductionOperation<std::uint64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<std::uint64_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceUnsigned8DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::uint64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint64_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceUnsigned8DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::uint64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::uint64_t *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<std::uint64_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#ifdef __SIZEOF_INT128__
common::uint128_t RTDEF(ReduceUnsigned16Ref)(const Descriptor &array,
    ReferenceReductionOperation<common::uint128_t> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const common::uint128_t *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<common::uint128_t, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
common::uint128_t RTDEF(ReduceUnsigned16Value)(const Descriptor &array,
    ValueReductionOperation<common::uint128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask,
    const common::uint128_t *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Unsigned, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<common::uint128_t, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceUnsigned16DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<common::uint128_t> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const common::uint128_t *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<common::uint128_t, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceUnsigned16DimValue)(Descriptor &result,
    const Descriptor &array,
    ValueReductionOperation<common::uint128_t> operation, const char *source,
    int line, int dim, const Descriptor *mask,
    const common::uint128_t *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<common::uint128_t, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Unsigned, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif

// TODO: real/complex(2 & 3)
float RTDEF(ReduceReal4Ref)(const Descriptor &array,
    ReferenceReductionOperation<float> operation, const char *source, int line,
    int dim, const Descriptor *mask, const float *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<float, false>{array, operation, identity, terminator},
      "REDUCE");
}
float RTDEF(ReduceReal4Value)(const Descriptor &array,
    ValueReductionOperation<float> operation, const char *source, int line,
    int dim, const Descriptor *mask, const float *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<float, true>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal4DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<float> operation, const char *source, int line,
    int dim, const Descriptor *mask, const float *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<float, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceReal4DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<float> operation, const char *source, int line,
    int dim, const Descriptor *mask, const float *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<float, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
double RTDEF(ReduceReal8Ref)(const Descriptor &array,
    ReferenceReductionOperation<double> operation, const char *source, int line,
    int dim, const Descriptor *mask, const double *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<double, false>{array, operation, identity, terminator},
      "REDUCE");
}
double RTDEF(ReduceReal8Value)(const Descriptor &array,
    ValueReductionOperation<double> operation, const char *source, int line,
    int dim, const Descriptor *mask, const double *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<double, true>{array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal8DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<double> operation, const char *source, int line,
    int dim, const Descriptor *mask, const double *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<double, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceReal8DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<double> operation, const char *source, int line,
    int dim, const Descriptor *mask, const double *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<double, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(ReduceReal10Ref)(
    const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Real, 10>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Real, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 10>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Real, 10>, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
CppTypeFor<TypeCategory::Real, 10> RTDEF(ReduceReal10Value)(
    const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Real, 10>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Real, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 10>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Real, 10>, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal10DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Real, 10>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Real, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Real, 10>, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 10>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceReal10DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Real, 10>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Real, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Real, 10>, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 10>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppFloat128Type RTDEF(ReduceReal16Ref)(const Descriptor &array,
    ReferenceReductionOperation<CppFloat128Type> operation, const char *source,
    int line, int dim, const Descriptor *mask, const CppFloat128Type *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppFloat128Type, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
CppFloat128Type RTDEF(ReduceReal16Value)(const Descriptor &array,
    ValueReductionOperation<CppFloat128Type> operation, const char *source,
    int line, int dim, const Descriptor *mask, const CppFloat128Type *identity,
    bool ordered) {
  Terminator terminator{source, line};
  return GetTotalReduction<TypeCategory::Real, 16>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppFloat128Type, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(ReduceReal16DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<CppFloat128Type> operation, const char *source,
    int line, int dim, const Descriptor *mask, const CppFloat128Type *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<CppFloat128Type, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(ReduceReal16DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<CppFloat128Type> operation, const char *source,
    int line, int dim, const Descriptor *mask, const CppFloat128Type *identity,
    bool ordered) {
  Terminator terminator{source, line};
  using Accumulator = ReduceAccumulator<CppFloat128Type, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Real, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif

void RTDEF(CppReduceComplex4Ref)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 4>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 4> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 4>, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex4Value)(CppTypeFor<TypeCategory::Complex, 4> &result,
    const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 4>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 4> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 4>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 4>, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex4DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 4>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 4> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 4>, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(CppReduceComplex4DimValue)(Descriptor &result,
    const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 4>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 4> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 4>, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 4>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(CppReduceComplex8Ref)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 8>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 8> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 8>, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex8Value)(CppTypeFor<TypeCategory::Complex, 8> &result,
    const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 8>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 8> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 8>(array, source, line, dim,
      mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 8>, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex8DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 8>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 8> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 8>, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(CppReduceComplex8DimValue)(Descriptor &result,
    const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 8>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 8> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 8>, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 8>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#if HAS_FLOAT80
void RTDEF(CppReduceComplex10Ref)(CppTypeFor<TypeCategory::Complex, 10> &result,
    const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 10>>
        operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 10>(array, source, line,
      dim, mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 10>, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex10Value)(
    CppTypeFor<TypeCategory::Complex, 10> &result, const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 10>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 10>(array, source, line,
      dim, mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 10>, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex10DimRef)(Descriptor &result,
    const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 10>>
        operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 10>, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 10>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(CppReduceComplex10DimValue)(Descriptor &result,
    const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 10>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 10> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 10>, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 10>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(CppReduceComplex16Ref)(CppTypeFor<TypeCategory::Complex, 16> &result,
    const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 16>>
        operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 16> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 16>(array, source, line,
      dim, mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 16>, false>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex16Value)(
    CppTypeFor<TypeCategory::Complex, 16> &result, const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 16>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 16> *identity, bool ordered) {
  Terminator terminator{source, line};
  result = GetTotalReduction<TypeCategory::Complex, 16>(array, source, line,
      dim, mask,
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 16>, true>{
          array, operation, identity, terminator},
      "REDUCE");
}
void RTDEF(CppReduceComplex16DimRef)(Descriptor &result,
    const Descriptor &array,
    ReferenceReductionOperation<CppTypeFor<TypeCategory::Complex, 16>>
        operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 16> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 16>, false>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
void RTDEF(CppReduceComplex16DimValue)(Descriptor &result,
    const Descriptor &array,
    ValueReductionOperation<CppTypeFor<TypeCategory::Complex, 16>> operation,
    const char *source, int line, int dim, const Descriptor *mask,
    const CppTypeFor<TypeCategory::Complex, 16> *identity, bool ordered) {
  Terminator terminator{source, line};
  using Accumulator =
      ReduceAccumulator<CppTypeFor<TypeCategory::Complex, 16>, true>;
  Accumulator accumulator{array, operation, identity, terminator};
  PartialReduction<Accumulator, TypeCategory::Complex, 16>(result, array,
      array.ElementBytes(), dim, mask, terminator, "REDUCE", accumulator);
}
#endif

bool RTDEF(ReduceLogical1Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger1Ref)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
bool RTDEF(ReduceLogical1Value)(const Descriptor &array,
    ValueReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger1Value)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical1DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger1DimRef)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
void RTDEF(ReduceLogical1DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int8_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int8_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger1DimValue)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
bool RTDEF(ReduceLogical2Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger2Ref)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
bool RTDEF(ReduceLogical2Value)(const Descriptor &array,
    ValueReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger2Value)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical2DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger2DimRef)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
void RTDEF(ReduceLogical2DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int16_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int16_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger2DimValue)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
bool RTDEF(ReduceLogical4Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger4Ref)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
bool RTDEF(ReduceLogical4Value)(const Descriptor &array,
    ValueReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger4Value)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical4DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger4DimRef)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
void RTDEF(ReduceLogical4DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int32_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int32_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger4DimValue)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
bool RTDEF(ReduceLogical8Ref)(const Descriptor &array,
    ReferenceReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger8Ref)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
bool RTDEF(ReduceLogical8Value)(const Descriptor &array,
    ValueReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  return RTNAME(ReduceInteger8Value)(
             array, operation, source, line, dim, mask, identity, ordered) != 0;
}
void RTDEF(ReduceLogical8DimRef)(Descriptor &result, const Descriptor &array,
    ReferenceReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger8DimRef)
  (result, array, operation, source, line, dim, mask, identity, ordered);
}
void RTDEF(ReduceLogical8DimValue)(Descriptor &result, const Descriptor &array,
    ValueReductionOperation<std::int64_t> operation, const char *source,
    int line, int dim, const Descriptor *mask, const std::int64_t *identity,
    bool ordered) {
  RTNAME(ReduceInteger8DimValue)
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
