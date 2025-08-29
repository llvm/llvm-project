//===-- lib/runtime/extrema.cpp ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements MAXLOC, MINLOC, MAXVAL, & MINVAL for all required operand types
// and shapes and (for MAXLOC & MINLOC) result integer kinds.  Also implements
// NORM2 using common infrastructure.

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

template <typename T, bool IS_MAX, bool BACK> struct NumericCompare {
  using Type = T;
  explicit RT_API_ATTRS NumericCompare(std::size_t /*elemLen; ignored*/) {}
  RT_API_ATTRS bool operator()(const T &value, const T &previous) const {
    if (std::is_floating_point_v<T> && previous != previous) {
      return BACK || value == value; // replace NaN
    } else if (value == previous) {
      return BACK;
    } else if constexpr (IS_MAX) {
      return value > previous;
    } else {
      return value < previous;
    }
  }
};

template <typename T, bool IS_MAX, bool BACK> class CharacterCompare {
public:
  using Type = T;
  explicit RT_API_ATTRS CharacterCompare(std::size_t elemLen)
      : chars_{elemLen / sizeof(T)} {}
  RT_API_ATTRS bool operator()(const T &value, const T &previous) const {
    int cmp{CharacterScalarCompare<T>(&value, &previous, chars_, chars_)};
    if (cmp == 0) {
      return BACK;
    } else if constexpr (IS_MAX) {
      return cmp > 0;
    } else {
      return cmp < 0;
    }
  }

private:
  std::size_t chars_;
};

template <typename COMPARE> class ExtremumLocAccumulator {
public:
  using Type = typename COMPARE::Type;
  RT_API_ATTRS ExtremumLocAccumulator(const Descriptor &array)
      : array_{array}, argRank_{array.rank()}, compare_{array.ElementBytes()} {
    Reinitialize();
  }
  RT_API_ATTRS void Reinitialize() {
    // per standard: result indices are all zero if no data
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
static RT_API_ATTRS void LocationHelper(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, const Descriptor *mask,
    Terminator &terminator) {
  ACCUMULATOR accumulator{x};
  DoTotalReduction<CPPTYPE>(x, 0, mask, accumulator, intrinsic, terminator);
  ApplyIntegerKind<LocationResultHelper<ACCUMULATOR>::template Functor, void>(
      kind, terminator, accumulator, result);
}

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
inline RT_API_ATTRS void DoMaxOrMinLoc(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, const char *source,
    int line, const Descriptor *mask, bool back) {
  using CppType = CppTypeFor<CAT, KIND>;
  Terminator terminator{source, line};
  if (back) {
    LocationHelper<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, true>>,
        CppType>(intrinsic, result, x, kind, mask, terminator);
  } else {
    LocationHelper<ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, false>>,
        CppType>(intrinsic, result, x, kind, mask, terminator);
  }
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

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(MaxlocCharacter)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  CharacterMaxOrMinLoc<true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 1, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 2, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 4, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocInteger8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 8, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MaxlocInteger16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 16, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MaxlocUnsigned1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 1, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocUnsigned2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 2, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocUnsigned4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 4, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocUnsigned8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 8, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MaxlocUnsigned16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 16, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MaxlocReal4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 4, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MaxlocReal8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 8, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#if HAS_FLOAT80
void RTDEF(MaxlocReal10)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 10, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(MaxlocReal16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 16, true>(
      "MAXLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MinlocCharacter)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  CharacterMaxOrMinLoc<false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 1, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 2, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 4, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocInteger8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 8, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MinlocInteger16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Integer, 16, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MinlocUnsigned1)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 1, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocUnsigned2)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 2, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocUnsigned4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 4, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocUnsigned8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 8, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#ifdef __SIZEOF_INT128__
void RTDEF(MinlocUnsigned16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Unsigned, 16, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif
void RTDEF(MinlocReal4)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 4, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
void RTDEF(MinlocReal8)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 8, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#if HAS_FLOAT80
void RTDEF(MinlocReal10)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 10, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
void RTDEF(MinlocReal16)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TotalNumericMaxOrMinLoc<TypeCategory::Real, 16, false>(
      "MINLOC", result, x, kind, source, line, mask, back);
}
#endif

RT_EXT_API_GROUP_END
} // extern "C"

// MAXLOC/MINLOC with DIM=

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE, bool BACK>
static RT_API_ATTRS void DoPartialMaxOrMinLocDirection(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, int dim,
    const Descriptor *mask, Terminator &terminator) {
  using CppType = CppTypeFor<CAT, KIND>;
  using Accumulator = ExtremumLocAccumulator<COMPARE<CppType, IS_MAX, BACK>>;
  Accumulator accumulator{x};
  ApplyIntegerKind<PartialLocationHelper<Accumulator>::template Functor, void>(
      kind, terminator, result, x, dim, mask, terminator, intrinsic,
      accumulator);
}

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
inline RT_API_ATTRS void DoPartialMaxOrMinLoc(const char *intrinsic,
    Descriptor &result, const Descriptor &x, int kind, int dim,
    const Descriptor *mask, bool back, Terminator &terminator) {
  if (back) {
    DoPartialMaxOrMinLocDirection<CAT, KIND, IS_MAX, COMPARE, true>(
        intrinsic, result, x, kind, dim, mask, terminator);
  } else {
    DoPartialMaxOrMinLocDirection<CAT, KIND, IS_MAX, COMPARE, false>(
        intrinsic, result, x, kind, dim, mask, terminator);
  }
}

template <TypeCategory CAT, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
struct DoPartialMaxOrMinLocHelper {
  template <int KIND> struct Functor {
    RT_API_ATTRS void operator()(const char *intrinsic, Descriptor &result,
        const Descriptor &x, int kind, int dim, const Descriptor *mask,
        bool back, Terminator &terminator) const {
      DoPartialMaxOrMinLoc<CAT, KIND, IS_MAX, COMPARE>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
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
      // A scalar MASK that's .TRUE.  In this case, just get rid of the MASK.
      maskToUse = nullptr;
    } else {
      // For scalar MASK arguments that are .FALSE., return all zeroes

      // Element size of the destination descriptor is the size
      // of {TypeCategory::Integer, kind}.
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

extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(MaxlocDim)(Descriptor &result, const Descriptor &x, int kind,
    int dim, const char *source, int line, const Descriptor *mask, bool back) {
  TypedPartialMaxOrMinLoc<true>(
      "MAXLOC", result, x, kind, dim, source, line, mask, back);
}
void RTDEF(MinlocDim)(Descriptor &result, const Descriptor &x, int kind,
    int dim, const char *source, int line, const Descriptor *mask, bool back) {
  TypedPartialMaxOrMinLoc<false>(
      "MINLOC", result, x, kind, dim, source, line, mask, back);
}

RT_EXT_API_GROUP_END
} // extern "C"

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
      // Empty array; fill with character 0 for MAXVAL.
      // For MINVAL, set all of the bits.
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

extern "C" {
RT_EXT_API_GROUP_BEGIN

CppTypeFor<TypeCategory::Integer, 1> RTDEF(MaxvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(MaxvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(MaxvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(MaxvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(MaxvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

CppTypeFor<TypeCategory::Unsigned, 1> RTDEF(MaxvalUnsigned1)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 1, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Unsigned, 2> RTDEF(MaxvalUnsigned2)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 2, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Unsigned, 4> RTDEF(MaxvalUnsigned4)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Unsigned, 8> RTDEF(MaxvalUnsigned8)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Unsigned, 16> RTDEF(MaxvalUnsigned16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(MaxvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(MaxvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(MaxvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(MaxvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

void RTDEF(MaxvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<true>(result, x, 0, source, line, mask, "MAXVAL");
}

CppTypeFor<TypeCategory::Integer, 1> RTDEF(MinvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(MinvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(MinvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(MinvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(MinvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

CppTypeFor<TypeCategory::Unsigned, 1> RTDEF(MinvalUnsigned1)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 1, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Unsigned, 2> RTDEF(MinvalUnsigned2)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 2, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Unsigned, 4> RTDEF(MinvalUnsigned4)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Unsigned, 8> RTDEF(MinvalUnsigned8)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Unsigned, 16> RTDEF(MinvalUnsigned16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Unsigned, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(MinvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(MinvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(MinvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif
#if HAS_LDBL128 || HAS_FLOAT128
CppTypeFor<TypeCategory::Real, 16> RTDEF(MinvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

void RTDEF(MinvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<false>(result, x, 0, source, line, mask, "MINVAL");
}

void RTDEF(MaxvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  } else {
    NumericMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  }
}
void RTDEF(MinvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  } else {
    NumericMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  }
}

RT_EXT_API_GROUP_END
} // extern "C"

// NORM2

extern "C" {
RT_EXT_API_GROUP_BEGIN

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTDEF(Norm2_4)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalReduction<TypeCategory::Real, 4>(
      x, source, line, dim, nullptr, Norm2Accumulator<4>{x}, "NORM2");
}
CppTypeFor<TypeCategory::Real, 8> RTDEF(Norm2_8)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalReduction<TypeCategory::Real, 8>(
      x, source, line, dim, nullptr, Norm2Accumulator<8>{x}, "NORM2");
}
#if HAS_FLOAT80
CppTypeFor<TypeCategory::Real, 10> RTDEF(Norm2_10)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalReduction<TypeCategory::Real, 10>(
      x, source, line, dim, nullptr, Norm2Accumulator<10>{x}, "NORM2");
}
#endif

void RTDEF(Norm2Dim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  if (type->first == TypeCategory::Real) {
    ApplyFloatingPointKind<Norm2Helper, void, true>(
        type->second, terminator, result, x, dim, nullptr, terminator);
  } else {
    terminator.Crash("NORM2: bad type code %d", x.type().raw());
  }
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
