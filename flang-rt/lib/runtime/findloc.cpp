//===-- lib/runtime/findloc.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements FINDLOC for all required operand types and shapes and result
// integer kinds.

#include "flang-rt/runtime/reduction-templates.h"
#include "flang/Runtime/character.h"
#include "flang/Runtime/reduction.h"
#include <cinttypes>
#include <complex>

namespace Fortran::runtime {

template <TypeCategory CAT1, int KIND1, TypeCategory CAT2, int KIND2>
struct Equality {
  using Type1 = CppTypeFor<CAT1, KIND1>;
  using Type2 = CppTypeFor<CAT2, KIND2>;
  RT_API_ATTRS bool operator()(const Descriptor &array,
      const SubscriptValue at[], const Descriptor &target) const {
    if constexpr (KIND1 >= KIND2) {
      return *array.Element<Type1>(at) ==
          static_cast<Type1>(*target.OffsetElement<Type2>());
    } else {
      return static_cast<Type2>(*array.Element<Type1>(at)) ==
          *target.OffsetElement<Type2>();
    }
  }
};

template <int KIND1, int KIND2>
struct Equality<TypeCategory::Complex, KIND1, TypeCategory::Complex, KIND2> {
  using Type1 = CppTypeFor<TypeCategory::Complex, KIND1>;
  using Type2 = CppTypeFor<TypeCategory::Complex, KIND2>;
  RT_API_ATTRS bool operator()(const Descriptor &array,
      const SubscriptValue at[], const Descriptor &target) const {
    const Type1 &xz{*array.Element<Type1>(at)};
    const Type2 &tz{*target.OffsetElement<Type2>()};
    return xz.real() == tz.real() && xz.imag() == tz.imag();
  }
};

template <int KIND1, TypeCategory CAT2, int KIND2>
struct Equality<TypeCategory::Complex, KIND1, CAT2, KIND2> {
  using Type1 = CppTypeFor<TypeCategory::Complex, KIND1>;
  using Type2 = CppTypeFor<CAT2, KIND2>;
  RT_API_ATTRS bool operator()(const Descriptor &array,
      const SubscriptValue at[], const Descriptor &target) const {
    const Type1 &z{*array.Element<Type1>(at)};
    return z.imag() == 0 && z.real() == *target.OffsetElement<Type2>();
  }
};

template <TypeCategory CAT1, int KIND1, int KIND2>
struct Equality<CAT1, KIND1, TypeCategory::Complex, KIND2> {
  using Type1 = CppTypeFor<CAT1, KIND1>;
  using Type2 = CppTypeFor<TypeCategory::Complex, KIND2>;
  RT_API_ATTRS bool operator()(const Descriptor &array,
      const SubscriptValue at[], const Descriptor &target) const {
    const Type2 &z{*target.OffsetElement<Type2>()};
    return *array.Element<Type1>(at) == z.real() && z.imag() == 0;
  }
};

template <int KIND> struct CharacterEquality {
  using Type = CppTypeFor<TypeCategory::Character, KIND>;
  RT_API_ATTRS bool operator()(const Descriptor &array,
      const SubscriptValue at[], const Descriptor &target) const {
    return CharacterScalarCompare<Type>(array.Element<Type>(at),
               target.OffsetElement<Type>(),
               array.ElementBytes() / static_cast<unsigned>(KIND),
               target.ElementBytes() / static_cast<unsigned>(KIND)) == 0;
  }
};

struct LogicalEquivalence {
  RT_API_ATTRS bool operator()(const Descriptor &array,
      const SubscriptValue at[], const Descriptor &target) const {
    return IsLogicalElementTrue(array, at) ==
        IsLogicalElementTrue(target, at /*ignored*/);
  }
};

template <TypeCategory CAT1, int KIND1, TypeCategory CAT2>
struct EqualityForTargetKind {
  template <int KIND2> struct Functor {
    RT_API_ATTRS void operator()(bool &result, const Descriptor &array,
        const SubscriptValue at[], const Descriptor &target) const {
      result = Equality<CAT1, KIND1, CAT2, KIND2>{}(array, at, target);
    }
  };
};

template <TypeCategory CAT, int KIND> class NumericFindlocAccumulator {
public:
  RT_API_ATTRS NumericFindlocAccumulator(const Descriptor &array,
      const Descriptor &target, bool back, TypeCategory targetCat,
      int targetKind, Terminator &terminator)
      : array_{array}, target_{target}, back_{back}, targetCat_{targetCat},
        targetKind_{targetKind}, terminator_{terminator} {}
  RT_API_ATTRS void Reinitialize() { gotAnything_ = false; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int zeroBasedDim = -1) {
    if (zeroBasedDim >= 0) {
      *p = gotAnything_ ? location_[zeroBasedDim] -
              array_.GetDimension(zeroBasedDim).LowerBound() + 1
                        : 0;
    } else if (gotAnything_) {
      for (int j{0}; j < rank_; ++j) {
        p[j] = location_[j] - array_.GetDimension(j).LowerBound() + 1;
      }
    } else {
      for (int j{0}; j < rank_; ++j) {
        p[j] = 0;
      }
    }
  }
  template <typename IGNORED>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    if (compareTarget(at)) {
      gotAnything_ = true;
      for (int j{0}; j < rank_; ++j) {
        location_[j] = at[j];
      }
      return back_;
    }
    return true;
  }

private:
  RT_API_ATTRS bool compareTarget(const SubscriptValue at[]) {
    bool result{false};
    switch (targetCat_) {
    case TypeCategory::Integer:
    case TypeCategory::Unsigned:
      ApplyIntegerKind<EqualityForTargetKind<CAT, KIND,
                           TypeCategory::Integer>::template Functor,
          void>(targetKind_, terminator_, result, array_, at, target_);
      break;
    case TypeCategory::Real:
      ApplyFloatingPointKind<EqualityForTargetKind<CAT, KIND,
                                 TypeCategory::Real>::template Functor,
          void>(targetKind_, terminator_, result, array_, at, target_);
      break;
    case TypeCategory::Complex:
      ApplyFloatingPointKind<EqualityForTargetKind<CAT, KIND,
                                 TypeCategory::Complex>::template Functor,
          void>(targetKind_, terminator_, result, array_, at, target_);
      break;
    default:
      break;
    }
    return result;
  }

  const Descriptor &array_;
  const Descriptor &target_;
  const bool back_{false};
  const int rank_{array_.rank()};
  bool gotAnything_{false};
  SubscriptValue location_[maxRank];
  const TypeCategory targetCat_;
  const int targetKind_;
  Terminator &terminator_;
};

template <typename EQUALITY> class LocationAccumulator {
public:
  RT_API_ATTRS LocationAccumulator(
      const Descriptor &array, const Descriptor &target, bool back)
      : array_{array}, target_{target}, back_{back} {}
  RT_API_ATTRS void Reinitialize() { gotAnything_ = false; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int zeroBasedDim = -1) {
    if (zeroBasedDim >= 0) {
      *p = gotAnything_ ? location_[zeroBasedDim] -
              array_.GetDimension(zeroBasedDim).LowerBound() + 1
                        : 0;
    } else if (gotAnything_) {
      for (int j{0}; j < rank_; ++j) {
        p[j] = location_[j] - array_.GetDimension(j).LowerBound() + 1;
      }
    } else {
      // no unmasked hits? result is all zeroes
      for (int j{0}; j < rank_; ++j) {
        p[j] = 0;
      }
    }
  }
  template <typename IGNORED>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    if (equality_(array_, at, target_)) {
      gotAnything_ = true;
      for (int j{0}; j < rank_; ++j) {
        location_[j] = at[j];
      }
      return back_;
    } else {
      return true;
    }
  }

private:
  const Descriptor &array_;
  const Descriptor &target_;
  const bool back_{false};
  const int rank_{array_.rank()};
  bool gotAnything_{false};
  SubscriptValue location_[maxRank];
  const EQUALITY equality_{};
};

template <TypeCategory CAT> struct TotalNumericFindlocSource {
  template <int KIND> struct Functor {
    RT_API_ATTRS RT_DEVICE_NOINLINE void operator()(TypeCategory targetCat,
        int targetKind, Descriptor &result, const Descriptor &x,
        const Descriptor &target, int kind, int dim, const Descriptor *mask,
        bool back, Terminator &terminator) const {
      using Accumulator = NumericFindlocAccumulator<CAT, KIND>;
      Accumulator accumulator{
          x, target, back, targetCat, targetKind, terminator};
      DoTotalReduction<void>(x, dim, mask, accumulator, "FINDLOC", terminator);
      ApplyIntegerKind<LocationResultHelper<Accumulator>::template Functor,
          void>(kind, terminator, accumulator, result);
    }
  };
};

template <int KIND> struct CharacterFindlocHelper {
  RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x,
      const Descriptor &target, int kind, const Descriptor *mask, bool back,
      Terminator &terminator) {
    using Accumulator = LocationAccumulator<CharacterEquality<KIND>>;
    Accumulator accumulator{x, target, back};
    DoTotalReduction<void>(x, 0, mask, accumulator, "FINDLOC", terminator);
    ApplyIntegerKind<LocationResultHelper<Accumulator>::template Functor, void>(
        kind, terminator, accumulator, result);
  }
};

static RT_API_ATTRS void LogicalFindlocHelper(Descriptor &result,
    const Descriptor &x, const Descriptor &target, int kind,
    const Descriptor *mask, bool back, Terminator &terminator) {
  using Accumulator = LocationAccumulator<LogicalEquivalence>;
  Accumulator accumulator{x, target, back};
  DoTotalReduction<void>(x, 0, mask, accumulator, "FINDLOC", terminator);
  ApplyIntegerKind<LocationResultHelper<Accumulator>::template Functor, void>(
      kind, terminator, accumulator, result);
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(Findloc)(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
  int rank{x.rank()};
  SubscriptValue extent[1]{rank};
  result.Establish(TypeCategory::Integer, kind, nullptr, 1, extent,
      CFI_attribute_allocatable);
  result.GetDimension(0).SetBounds(1, extent[0]);
  Terminator terminator{source, line};
  if (int stat{result.Allocate(kNoAsyncObject)}) {
    terminator.Crash(
        "FINDLOC: could not allocate memory for result; STAT=%d", stat);
  }
  CheckIntegerKind(terminator, kind, "FINDLOC");
  auto xType{x.type().GetCategoryAndKind()};
  auto targetType{target.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, xType.has_value() && targetType.has_value());
  switch (xType->first) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
    ApplyIntegerKind<
        TotalNumericFindlocSource<TypeCategory::Integer>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        TotalNumericFindlocSource<TypeCategory::Real>::template Functor, void>(
        xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<
        TotalNumericFindlocSource<TypeCategory::Complex>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, 0, mask, back, terminator);
    break;
  case TypeCategory::Character:
    RUNTIME_CHECK(terminator,
        targetType->first == TypeCategory::Character &&
            targetType->second == xType->second);
    ApplyCharacterKind<CharacterFindlocHelper, void>(xType->second, terminator,
        result, x, target, kind, mask, back, terminator);
    break;
  case TypeCategory::Logical:
    RUNTIME_CHECK(terminator, targetType->first == TypeCategory::Logical);
    LogicalFindlocHelper(result, x, target, kind, mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "FINDLOC: bad data type code (%d) for array", x.type().raw());
  }
}

RT_EXT_API_GROUP_END
} // extern "C"

// FINDLOC with DIM=

template <TypeCategory CAT> struct PartialNumericFindlocSource {
  template <int KIND> struct Functor {
    RT_API_ATTRS RT_DEVICE_NOINLINE void operator()(TypeCategory targetCat,
        int targetKind, Descriptor &result, const Descriptor &x,
        const Descriptor &target, int kind, int dim, const Descriptor *mask,
        bool back, Terminator &terminator) const {
      using Accumulator = NumericFindlocAccumulator<CAT, KIND>;
      Accumulator accumulator{
          x, target, back, targetCat, targetKind, terminator};
      ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor,
          void>(kind, terminator, result, x, dim, mask, terminator, "FINDLOC",
          accumulator);
    }
  };
};

template <int KIND> struct PartialCharacterFindlocHelper {
  RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x,
      const Descriptor &target, int kind, int dim, const Descriptor *mask,
      bool back, Terminator &terminator) {
    using Accumulator = LocationAccumulator<CharacterEquality<KIND>>;
    Accumulator accumulator{x, target, back};
    ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor,
        void>(kind, terminator, result, x, dim, mask, terminator, "FINDLOC",
        accumulator);
  }
};

static RT_API_ATTRS void PartialLogicalFindlocHelper(Descriptor &result,
    const Descriptor &x, const Descriptor &target, int kind, int dim,
    const Descriptor *mask, bool back, Terminator &terminator) {
  using Accumulator = LocationAccumulator<LogicalEquivalence>;
  Accumulator accumulator{x, target, back};
  ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor, void>(
      kind, terminator, result, x, dim, mask, terminator, "FINDLOC",
      accumulator);
}

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(FindlocDim)(Descriptor &result, const Descriptor &x,
    const Descriptor &target, int kind, int dim, const char *source, int line,
    const Descriptor *mask, bool back) {
  Terminator terminator{source, line};
  CheckIntegerKind(terminator, kind, "FINDLOC");
  auto xType{x.type().GetCategoryAndKind()};
  auto targetType{target.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, xType.has_value() && targetType.has_value());
  switch (xType->first) {
  case TypeCategory::Integer:
  case TypeCategory::Unsigned:
    ApplyIntegerKind<
        PartialNumericFindlocSource<TypeCategory::Integer>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        PartialNumericFindlocSource<TypeCategory::Real>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Complex:
    ApplyFloatingPointKind<
        PartialNumericFindlocSource<TypeCategory::Complex>::template Functor,
        void>(xType->second, terminator, targetType->first, targetType->second,
        result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Character:
    RUNTIME_CHECK(terminator,
        targetType->first == TypeCategory::Character &&
            targetType->second == xType->second);
    ApplyCharacterKind<PartialCharacterFindlocHelper, void>(xType->second,
        terminator, result, x, target, kind, dim, mask, back, terminator);
    break;
  case TypeCategory::Logical:
    RUNTIME_CHECK(terminator, targetType->first == TypeCategory::Logical);
    PartialLogicalFindlocHelper(
        result, x, target, kind, dim, mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "FINDLOC: bad data type code (%d) for array", x.type().raw());
  }
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
