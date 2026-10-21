//===-- lib/runtime/extrema.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements MAXLOC, MINLOC, MAXVAL, & MINVAL for all required operand types
// and shapes and (for MAXLOC & MINLOC) result integer kinds.  Also implements
// NORM2 using common infrastructure.

#ifndef FLANG_RT_RUNTIME_EXTREMA_H_
#define FLANG_RT_RUNTIME_EXTREMA_H_

#include "flang-rt/runtime/reduction-templates.h"
#include "flang/Common/float128.h"
#include "flang/Runtime/character.h"
#include "flang/Runtime/reduction.h"
#include <algorithm>
#include <cfloat>
#include <cinttypes>
#include <cmath>
#include <type_traits>

namespace Fortran::runtime {

// MAXLOC & MINLOC

template <typename T, bool IS_MAX> struct NumericCompare {
  using Type = T;
  explicit RT_API_ATTRS NumericCompare(std::size_t /*elemLen*/, bool back)
      : back_{back} {}
  RT_API_ATTRS bool operator()(const T &value, const T &previous) const {
    if (std::is_floating_point_v<T> && previous != previous) {
      return back_ || value == value; // replace NaN
    } else if (value == previous) {
      return back_;
    } else if constexpr (IS_MAX) {
      return value > previous;
    } else {
      return value < previous;
    }
  }

private:
  bool back_;
};

template <typename T, bool IS_MAX> class CharacterCompare {
public:
  using Type = T;
  explicit RT_API_ATTRS CharacterCompare(std::size_t elemLen, bool back)
      : chars_{elemLen / sizeof(T)}, back_{back} {}
  RT_API_ATTRS bool operator()(const T &value, const T &previous) const {
    int cmp{CharacterScalarCompare<T>(&value, &previous, chars_, chars_)};
    if (cmp == 0) {
      return back_;
    } else if constexpr (IS_MAX) {
      return cmp > 0;
    } else {
      return cmp < 0;
    }
  }

private:
  std::size_t chars_;
  bool back_;
};

template <typename COMPARE> class ExtremumLocAccumulator {
public:
  using Type = typename COMPARE::Type;
  RT_API_ATTRS ExtremumLocAccumulator(const Descriptor &array, bool back)
      : array_{array}, argRank_{array.rank()},
        compare_{array.ElementBytes(), back} {
    Reinitialize();
  }
  RT_API_ATTRS void Reinitialize() {
    for (int j{0}; j < argRank_; ++j) {
      extremumLoc_[j] = 0;
    }
    previous_ = nullptr;
  }
  RT_API_ATTRS int argRank() const { return argRank_; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int zeroBasedDim = -1) {
    if (zeroBasedDim >= 0) {
      *p = extremumLoc_[zeroBasedDim];
    } else {
      for (int j{0}; j < argRank_; ++j) {
        p[j] = extremumLoc_[j];
      }
    }
  }
  template <typename IGNORED>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    const auto &value{*array_.Element<Type>(at)};
    if (!previous_ || compare_(value, *previous_)) {
      previous_ = &value;
      for (int j{0}; j < argRank_; ++j) {
        extremumLoc_[j] = at[j] - array_.GetDimension(j).LowerBound() + 1;
      }
    }
    return true;
  }

private:
  const Descriptor &array_;
  int argRank_;
  SubscriptValue extremumLoc_[maxRank];
  const Type *previous_{nullptr};
  COMPARE compare_;
};

template <typename ACCUMULATOR, typename CPPTYPE>
inline RT_API_ATTRS void LocationHelper(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, const Descriptor *mask,
    Terminator &terminator, bool back) {
  ACCUMULATOR accumulator{x, back};
  DoTotalReduction<CPPTYPE>(x, 0, mask, accumulator, intrinsic, terminator);
  ApplyIntegerKind<LocationResultHelper<ACCUMULATOR>::template Functor, void>(
      kind, terminator, accumulator, result);
}

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool> class COMPARE>
inline RT_API_ATTRS void DoMaxOrMinLoc(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, const char *source,
    int line, const Descriptor *mask, bool back) {
  using CppType = CppTypeFor<CAT, KIND>;
  Terminator terminator{source, line};
  LocationHelper<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX>>, CppType>(
      intrinsic, result, x, kind, mask, terminator, back);
}

template <bool IS_MAX> struct CharacterMaxOrMinLocHelper {
  template <int KIND> struct Functor {
    RT_API_ATTRS void operator()(const char *intrinsic, Descriptor &result,
        const Descriptor &x, int kind, const char *source, int line,
        const Descriptor *mask, bool back) const {
      DoMaxOrMinLoc<TypeCategory::Character, KIND, IS_MAX, CharacterCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
    }
  };
};

template <bool IS_MAX>
inline RT_API_ATTRS void CharacterMaxOrMinLoc(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, const char *source,
    int line, const Descriptor *mask, bool back) {
  int rank{x.rank()};
  SubscriptValue extent[1]{rank};
  result.Establish(TypeCategory::Integer, kind, nullptr, 1, extent,
      CFI_attribute_allocatable);
  result.GetDimension(0).SetBounds(1, extent[0]);
  Terminator terminator{source, line};
  if (int stat{result.Allocate(kNoAsyncObject)}) {
    terminator.Crash(
        "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
  }
  CheckIntegerKind(terminator, kind, intrinsic);
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind.has_value());
  switch (catKind->first) {
  case TypeCategory::Character:
    ApplyCharacterKind<CharacterMaxOrMinLocHelper<IS_MAX>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, source,
        line, mask, back);
    break;
  default:
    terminator.Crash(
        "%s: bad data type code (%d) for array", intrinsic, x.type().raw());
  }
}

template <TypeCategory CAT, int KIND, bool IS_MAXVAL>
inline RT_API_ATTRS void TotalNumericMaxOrMinLoc(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, const char *source,
    int line, const Descriptor *mask, bool back) {
  int rank{x.rank()};
  SubscriptValue extent[1]{rank};
  result.Establish(TypeCategory::Integer, kind, nullptr, 1, extent,
      CFI_attribute_allocatable);
  result.GetDimension(0).SetBounds(1, extent[0]);
  Terminator terminator{source, line};
  if (int stat{result.Allocate(kNoAsyncObject)}) {
    terminator.Crash(
        "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
  }
  CheckIntegerKind(terminator, kind, intrinsic);
  RUNTIME_CHECK(terminator, TypeCode(CAT, KIND) == x.type());
  DoMaxOrMinLoc<CAT, KIND, IS_MAXVAL, NumericCompare>(
      intrinsic, result, x, kind, source, line, mask, back);
}

// MAXLOC/MINLOC with DIM=

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool> class COMPARE>
inline RT_API_ATTRS void DoPartialMaxOrMinLoc(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, int dim,
    const Descriptor *mask, Terminator &terminator, bool back) {
  using CppType = CppTypeFor<CAT, KIND>;
  using Accumulator = ExtremumLocAccumulator<COMPARE<CppType, IS_MAX>>;
  Accumulator accumulator{x, back};
  ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor, void>(
      kind, terminator, result, x, dim, mask, terminator, intrinsic,
      accumulator);
}

template <TypeCategory CAT, bool IS_MAX,
    template <typename, bool> class COMPARE>
struct DoPartialMaxOrMinLocHelper {
  template <int KIND> struct Functor {
    RT_API_ATTRS RT_DEVICE_NOINLINE void operator()(const char *intrinsic,
        Descriptor &result, const Descriptor &x, int kind, int dim,
        const Descriptor *mask, bool back, Terminator &terminator) const {
      DoPartialMaxOrMinLoc<CAT, KIND, IS_MAX, COMPARE>(
          intrinsic, result, x, kind, dim, mask, terminator, back);
    }
  };
};

template <bool IS_MAX>
inline RT_API_ATTRS void TypedPartialMaxOrMinLoc(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, int dim,
    const char *source, int line, const Descriptor *mask, bool back) {
  Terminator terminator{source, line};
  CheckIntegerKind(terminator, kind, intrinsic);
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind.has_value());
  const Descriptor *maskToUse{mask};
  SubscriptValue maskAt[maxRank]; // contents unused
  if (mask && mask->rank() == 0) {
    if (IsLogicalElementTrue(*mask, maskAt)) {
      maskToUse = nullptr;
    } else {
      CreatePartialReductionResult(result, x,
          Descriptor::BytesFor(TypeCategory::Integer, kind), dim, terminator,
          intrinsic, TypeCode{TypeCategory::Integer, kind});
      runtime::memset(
          result.OffsetElement(), 0, result.Elements() * result.ElementBytes());
      return;
    }
  }
  switch (catKind->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<DoPartialMaxOrMinLocHelper<TypeCategory::Integer, IS_MAX,
                         NumericCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        maskToUse, back, terminator);
    break;
  case TypeCategory::Unsigned:
    ApplyIntegerKind<DoPartialMaxOrMinLocHelper<TypeCategory::Unsigned, IS_MAX,
                         NumericCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        maskToUse, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<DoPartialMaxOrMinLocHelper<TypeCategory::Real,
                               IS_MAX, NumericCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        maskToUse, back, terminator);
    break;
  case TypeCategory::Character:
    ApplyCharacterKind<DoPartialMaxOrMinLocHelper<TypeCategory::Character,
                           IS_MAX, CharacterCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        maskToUse, back, terminator);
    break;
  default:
    terminator.Crash(
        "%s: bad data type code (%d) for array", intrinsic, x.type().raw());
  }
}

// MAXVAL and MINVAL

template <TypeCategory CAT, int KIND, bool IS_MAXVAL>
class NumericExtremumAccumulator {
public:
  using Type = CppTypeFor<CAT, KIND>;
  explicit RT_API_ATTRS NumericExtremumAccumulator(const Descriptor &array)
      : array_{array} {}
  RT_API_ATTRS void Reinitialize() {
    any_ = false;
    extremum_ = MaxOrMinIdentity<CAT, KIND, IS_MAXVAL>::Value();
  }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = extremum_;
  }
  RT_API_ATTRS bool Accumulate(Type x) {
    if (!any_) {
      extremum_ = x;
      any_ = true;
    } else if (CAT == TypeCategory::Real && extremum_ != extremum_) {
      extremum_ = x; // replace NaN
    } else if constexpr (IS_MAXVAL) {
      if (x > extremum_) {
        extremum_ = x;
      }
    } else if (x < extremum_) {
      extremum_ = x;
    }
    return true;
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  bool any_{false};
  Type extremum_{MaxOrMinIdentity<CAT, KIND, IS_MAXVAL>::Value()};
};

template <TypeCategory CAT, int KIND, bool IS_MAXVAL>
inline RT_API_ATTRS CppTypeFor<CAT, KIND> TotalNumericMaxOrMin(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask, const char *intrinsic) {
  return GetTotalReduction<CAT, KIND>(x, source, line, dim, mask,
      NumericExtremumAccumulator<CAT, KIND, IS_MAXVAL>{x}, intrinsic);
}

template <TypeCategory CAT, bool IS_MAXVAL> struct MaxOrMinHelper {
  template <int KIND> struct Functor {
    RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x,
        int dim, const Descriptor *mask, const char *intrinsic,
        Terminator &terminator) const {
      DoMaxMinNorm2<CAT, KIND,
          NumericExtremumAccumulator<CAT, KIND, IS_MAXVAL>>(
          result, x, dim, mask, intrinsic, terminator);
    }
  };
};

template <bool IS_MAXVAL>
inline RT_API_ATTRS void NumericMaxOrMin(Descriptor &result,
    const Descriptor &x, int dim, const char *source, int line,
    const Descriptor *mask, const char *intrinsic) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  switch (type->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<
        MaxOrMinHelper<TypeCategory::Integer, IS_MAXVAL>::template Functor,
        void>(
        type->second, terminator, result, x, dim, mask, intrinsic, terminator);
    break;
  case TypeCategory::Unsigned:
    ApplyIntegerKind<
        MaxOrMinHelper<TypeCategory::Unsigned, IS_MAXVAL>::template Functor,
        void>(
        type->second, terminator, result, x, dim, mask, intrinsic, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        MaxOrMinHelper<TypeCategory::Real, IS_MAXVAL>::template Functor, void>(
        type->second, terminator, result, x, dim, mask, intrinsic, terminator);
    break;
  default:
    terminator.Crash("%s: bad type code %d", intrinsic, x.type().raw());
  }
}

template <int KIND, bool IS_MAXVAL> class CharacterExtremumAccumulator {
public:
  using Type = CppTypeFor<TypeCategory::Character, KIND>;
  explicit RT_API_ATTRS CharacterExtremumAccumulator(const Descriptor &array)
      : array_{array}, charLen_{array_.ElementBytes() / KIND} {}
  RT_API_ATTRS void Reinitialize() { extremum_ = nullptr; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    static_assert(std::is_same_v<A, Type>);
    std::size_t byteSize{array_.ElementBytes()};
    if (extremum_) {
      runtime::memcpy(p, extremum_, byteSize);
    } else {
      runtime::memset(p, IS_MAXVAL ? 0 : 255, byteSize);
    }
  }
  RT_API_ATTRS bool Accumulate(const Type *x) {
    if (!extremum_) {
      extremum_ = x;
    } else {
      int cmp{CharacterScalarCompare(x, extremum_, charLen_, charLen_)};
      if (IS_MAXVAL == (cmp > 0)) {
        extremum_ = x;
      }
    }
    return true;
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  std::size_t charLen_;
  const Type *extremum_{nullptr};
};

template <bool IS_MAXVAL> struct CharacterMaxOrMinHelper {
  template <int KIND> struct Functor {
    RT_API_ATTRS void operator()(Descriptor &result, const Descriptor &x,
        int dim, const Descriptor *mask, const char *intrinsic,
        Terminator &terminator) const {
      DoMaxMinNorm2<TypeCategory::Character, KIND,
          CharacterExtremumAccumulator<KIND, IS_MAXVAL>>(
          result, x, dim, mask, intrinsic, terminator);
    }
  };
};

template <bool IS_MAXVAL>
inline RT_API_ATTRS void CharacterMaxOrMin(Descriptor &result,
    const Descriptor &x, int dim, const char *source, int line,
    const Descriptor *mask, const char *intrinsic) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type && type->first == TypeCategory::Character);
  ApplyCharacterKind<CharacterMaxOrMinHelper<IS_MAXVAL>::template Functor,
      void>(
      type->second, terminator, result, x, dim, mask, intrinsic, terminator);
}

} // namespace Fortran::runtime

#endif // FLANG_RT_RUNTIME_EXTREMA_H_
