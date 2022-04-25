//===-- runtime/extrema.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements MAXLOC, MINLOC, MAXVAL, & MINVAL for all required operand types
// and shapes and (for MAXLOC & MINLOC) result integer kinds.  Also implements
// NORM2 using common infrastructure.

#include "reduction-templates.h"
#include "flang/Common/long-double.h"
#include "flang/Runtime/character.h"
#include "flang/Runtime/reduction.h"
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <optional>

namespace Fortran::runtime {

// MAXLOC & MINLOC

template <typename T, bool IS_MAX, bool BACK> struct NumericCompare {
  using Type = T;
  explicit NumericCompare(std::size_t /*elemLen; ignored*/) {}
  bool operator()(const T &value, const T &previous) const {
    if (value == previous) {
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
  explicit CharacterCompare(std::size_t elemLen)
      : chars_{elemLen / sizeof(T)} {}
  bool operator()(const T &value, const T &previous) const {
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
  ExtremumLocAccumulator(const Descriptor &array, std::size_t chars = 0)
      : array_{array}, argRank_{array.rank()}, compare_{array.ElementBytes()} {
    Reinitialize();
  }
  void Reinitialize() {
    // per standard: result indices are all zero if no data
    for (int j{0}; j < argRank_; ++j) {
      extremumLoc_[j] = 0;
    }
    previous_ = nullptr;
  }
  int argRank() const { return argRank_; }
  template <typename A> void GetResult(A *p, int zeroBasedDim = -1) {
    if (zeroBasedDim >= 0) {
      *p = extremumLoc_[zeroBasedDim] -
          array_.GetDimension(zeroBasedDim).LowerBound() + 1;
    } else {
      for (int j{0}; j < argRank_; ++j) {
        p[j] = extremumLoc_[j] - array_.GetDimension(j).LowerBound() + 1;
      }
    }
  }
  template <typename IGNORED> bool AccumulateAt(const SubscriptValue at[]) {
    const auto &value{*array_.Element<Type>(at)};
    if (!previous_ || compare_(value, *previous_)) {
      previous_ = &value;
      for (int j{0}; j < argRank_; ++j) {
        extremumLoc_[j] = at[j];
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
static void LocationHelper(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const Descriptor *mask,
    Terminator &terminator) {
  ACCUMULATOR accumulator{x};
  DoTotalReduction<CPPTYPE>(x, 0, mask, accumulator, intrinsic, terminator);
  ApplyIntegerKind<LocationResultHelper<ACCUMULATOR>::template Functor, void>(
      kind, terminator, accumulator, result);
}

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE>
inline void DoMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
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

template <TypeCategory CAT, bool IS_MAX> struct TypedMaxOrMinLocHelper {
  template <int KIND> struct Functor {
    void operator()(const char *intrinsic, Descriptor &result,
        const Descriptor &x, int kind, const char *source, int line,
        const Descriptor *mask, bool back) const {
      DoMaxOrMinLoc<CAT, KIND, IS_MAX, NumericCompare>(
          intrinsic, result, x, kind, source, line, mask, back);
    }
  };
};

template <bool IS_MAX>
inline void TypedMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, const char *source, int line,
    const Descriptor *mask, bool back) {
  int rank{x.rank()};
  SubscriptValue extent[1]{rank};
  result.Establish(TypeCategory::Integer, kind, nullptr, 1, extent,
      CFI_attribute_allocatable);
  result.GetDimension(0).SetBounds(1, extent[0]);
  Terminator terminator{source, line};
  if (int stat{result.Allocate()}) {
    terminator.Crash(
        "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
  }
  CheckIntegerKind(terminator, kind, intrinsic);
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind.has_value());
  switch (catKind->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<
        TypedMaxOrMinLocHelper<TypeCategory::Integer, IS_MAX>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, source,
        line, mask, back);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<
        TypedMaxOrMinLocHelper<TypeCategory::Real, IS_MAX>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, source,
        line, mask, back);
    break;
  case TypeCategory::Character:
    ApplyCharacterKind<TypedMaxOrMinLocHelper<TypeCategory::Character,
                           IS_MAX>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, source,
        line, mask, back);
    break;
  default:
    terminator.Crash(
        "%s: bad data type code (%d) for array", intrinsic, x.type().raw());
  }
}

extern "C" {
void RTNAME(Maxloc)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TypedMaxOrMinLoc<true>("MAXLOC", result, x, kind, source, line, mask, back);
}
void RTNAME(Minloc)(Descriptor &result, const Descriptor &x, int kind,
    const char *source, int line, const Descriptor *mask, bool back) {
  TypedMaxOrMinLoc<false>("MINLOC", result, x, kind, source, line, mask, back);
}
} // extern "C"

// MAXLOC/MINLOC with DIM=

template <TypeCategory CAT, int KIND, bool IS_MAX,
    template <typename, bool, bool> class COMPARE, bool BACK>
static void DoPartialMaxOrMinLocDirection(const char *intrinsic,
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
inline void DoPartialMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, int dim, const Descriptor *mask, bool back,
    Terminator &terminator) {
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
    void operator()(const char *intrinsic, Descriptor &result,
        const Descriptor &x, int kind, int dim, const Descriptor *mask,
        bool back, Terminator &terminator) const {
      DoPartialMaxOrMinLoc<CAT, KIND, IS_MAX, COMPARE>(
          intrinsic, result, x, kind, dim, mask, back, terminator);
    }
  };
};

template <bool IS_MAX>
inline void TypedPartialMaxOrMinLoc(const char *intrinsic, Descriptor &result,
    const Descriptor &x, int kind, int dim, const char *source, int line,
    const Descriptor *mask, bool back) {
  Terminator terminator{source, line};
  CheckIntegerKind(terminator, kind, intrinsic);
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind.has_value());
  switch (catKind->first) {
  case TypeCategory::Integer:
    ApplyIntegerKind<DoPartialMaxOrMinLocHelper<TypeCategory::Integer, IS_MAX,
                         NumericCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        mask, back, terminator);
    break;
  case TypeCategory::Real:
    ApplyFloatingPointKind<DoPartialMaxOrMinLocHelper<TypeCategory::Real,
                               IS_MAX, NumericCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        mask, back, terminator);
    break;
  case TypeCategory::Character:
    ApplyCharacterKind<DoPartialMaxOrMinLocHelper<TypeCategory::Character,
                           IS_MAX, CharacterCompare>::template Functor,
        void>(catKind->second, terminator, intrinsic, result, x, kind, dim,
        mask, back, terminator);
    break;
  default:
    terminator.Crash(
        "%s: bad data type code (%d) for array", intrinsic, x.type().raw());
  }
}

extern "C" {
void RTNAME(MaxlocDim)(Descriptor &result, const Descriptor &x, int kind,
    int dim, const char *source, int line, const Descriptor *mask, bool back) {
  TypedPartialMaxOrMinLoc<true>(
      "MAXLOC", result, x, kind, dim, source, line, mask, back);
}
void RTNAME(MinlocDim)(Descriptor &result, const Descriptor &x, int kind,
    int dim, const char *source, int line, const Descriptor *mask, bool back) {
  TypedPartialMaxOrMinLoc<false>(
      "MINLOC", result, x, kind, dim, source, line, mask, back);
}
} // extern "C"

// MAXVAL and MINVAL

template <TypeCategory CAT, int KIND, bool IS_MAXVAL> struct MaxOrMinIdentity {
  using Type = CppTypeFor<CAT, KIND>;
  static constexpr Type Value() {
    return IS_MAXVAL ? std::numeric_limits<Type>::lowest()
                     : std::numeric_limits<Type>::max();
  }
};

// std::numeric_limits<> may not know int128_t
template <bool IS_MAXVAL>
struct MaxOrMinIdentity<TypeCategory::Integer, 16, IS_MAXVAL> {
  using Type = CppTypeFor<TypeCategory::Integer, 16>;
  static constexpr Type Value() {
    return IS_MAXVAL ? Type{1} << 127 : ~Type{0} >> 1;
  }
};

template <TypeCategory CAT, int KIND, bool IS_MAXVAL>
class NumericExtremumAccumulator {
public:
  using Type = CppTypeFor<CAT, KIND>;
  explicit NumericExtremumAccumulator(const Descriptor &array)
      : array_{array} {}
  void Reinitialize() {
    extremum_ = MaxOrMinIdentity<CAT, KIND, IS_MAXVAL>::Value();
  }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = extremum_;
  }
  bool Accumulate(Type x) {
    if constexpr (IS_MAXVAL) {
      if (x > extremum_) {
        extremum_ = x;
      }
    } else if (x < extremum_) {
      extremum_ = x;
    }
    return true;
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  Type extremum_{MaxOrMinIdentity<CAT, KIND, IS_MAXVAL>::Value()};
};

template <TypeCategory CAT, int KIND, bool IS_MAXVAL>
inline CppTypeFor<CAT, KIND> TotalNumericMaxOrMin(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask,
    const char *intrinsic) {
  return GetTotalReduction<CAT, KIND>(x, source, line, dim, mask,
      NumericExtremumAccumulator<CAT, KIND, IS_MAXVAL>{x}, intrinsic);
}

template <TypeCategory CAT, int KIND, typename ACCUMULATOR>
static void DoMaxMinNorm2(Descriptor &result, const Descriptor &x, int dim,
    const Descriptor *mask, const char *intrinsic, Terminator &terminator) {
  using Type = CppTypeFor<CAT, KIND>;
  ACCUMULATOR accumulator{x};
  if (dim == 0 || x.rank() == 1) {
    // Total reduction
    result.Establish(x.type(), x.ElementBytes(), nullptr, 0, nullptr,
        CFI_attribute_allocatable);
    if (int stat{result.Allocate()}) {
      terminator.Crash(
          "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
    }
    DoTotalReduction<Type>(x, dim, mask, accumulator, intrinsic, terminator);
    accumulator.GetResult(result.OffsetElement<Type>());
  } else {
    // Partial reduction
    PartialReduction<ACCUMULATOR, CAT, KIND>(
        result, x, dim, mask, terminator, intrinsic, accumulator);
  }
}

template <TypeCategory CAT, bool IS_MAXVAL> struct MaxOrMinHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, const char *intrinsic,
        Terminator &terminator) const {
      DoMaxMinNorm2<CAT, KIND,
          NumericExtremumAccumulator<CAT, KIND, IS_MAXVAL>>(
          result, x, dim, mask, intrinsic, terminator);
    }
  };
};

template <bool IS_MAXVAL>
inline void NumericMaxOrMin(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask,
    const char *intrinsic) {
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
  explicit CharacterExtremumAccumulator(const Descriptor &array)
      : array_{array}, charLen_{array_.ElementBytes() / KIND} {}
  void Reinitialize() { extremum_ = nullptr; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    static_assert(std::is_same_v<A, Type>);
    if (extremum_) {
      std::memcpy(p, extremum_, charLen_);
    } else {
      // empty array: result is all zero-valued characters
      std::memset(p, 0, charLen_);
    }
  }
  bool Accumulate(const Type *x) {
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
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  std::size_t charLen_;
  const Type *extremum_{nullptr};
};

template <bool IS_MAXVAL> struct CharacterMaxOrMinHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        const Descriptor *mask, const char *intrinsic,
        Terminator &terminator) const {
      DoMaxMinNorm2<TypeCategory::Character, KIND,
          CharacterExtremumAccumulator<KIND, IS_MAXVAL>>(
          result, x, dim, mask, intrinsic, terminator);
    }
  };
};

template <bool IS_MAXVAL>
inline void CharacterMaxOrMin(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask,
    const char *intrinsic) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type && type->first == TypeCategory::Character);
  ApplyCharacterKind<CharacterMaxOrMinHelper<IS_MAXVAL>::template Functor,
      void>(
      type->second, terminator, result, x, dim, mask, intrinsic, terminator);
}

extern "C" {
CppTypeFor<TypeCategory::Integer, 1> RTNAME(MaxvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(MaxvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(MaxvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(MaxvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(MaxvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(MaxvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, true>(
      x, source, line, dim, mask, "MAXVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(MaxvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(MaxvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(MaxvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, true>(
      x, source, line, dim, mask, "MAXVAL");
}
#endif

void RTNAME(MaxvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<true>(result, x, 0, source, line, mask, "MAXVAL");
}

CppTypeFor<TypeCategory::Integer, 1> RTNAME(MinvalInteger1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 1, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 2> RTNAME(MinvalInteger2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 2, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 4> RTNAME(MinvalInteger4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Integer, 8> RTNAME(MinvalInteger8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTNAME(MinvalInteger16)(
    const Descriptor &x, const char *source, int line, int dim,
    const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Integer, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(MinvalReal4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 4, false>(
      x, source, line, dim, mask, "MINVAL");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(MinvalReal8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 8, false>(
      x, source, line, dim, mask, "MINVAL");
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(MinvalReal10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 10, false>(
      x, source, line, dim, mask, "MINVAL");
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(MinvalReal16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return TotalNumericMaxOrMin<TypeCategory::Real, 16, false>(
      x, source, line, dim, mask, "MINVAL");
}
#endif

void RTNAME(MinvalCharacter)(Descriptor &result, const Descriptor &x,
    const char *source, int line, const Descriptor *mask) {
  CharacterMaxOrMin<false>(result, x, 0, source, line, mask, "MINVAL");
}

void RTNAME(MaxvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  } else {
    NumericMaxOrMin<true>(result, x, dim, source, line, mask, "MAXVAL");
  }
}
void RTNAME(MinvalDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  if (x.type().IsCharacter()) {
    CharacterMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  } else {
    NumericMaxOrMin<false>(result, x, dim, source, line, mask, "MINVAL");
  }
}
} // extern "C"

// NORM2

template <int KIND> class Norm2Accumulator {
public:
  using Type = CppTypeFor<TypeCategory::Real, KIND>;
  // Use at least double precision for accumulators
  using AccumType = CppTypeFor<TypeCategory::Real, std::max(KIND, 8)>;
  explicit Norm2Accumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { max_ = sum_ = 0; }
  template <typename A> void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    // m * sqrt(1 + sum((others(:)/m)**2))
    *p = static_cast<Type>(max_ * std::sqrt(1 + sum_));
  }
  bool Accumulate(Type x) {
    auto absX{AccumType{std::abs(x)}};
    if (!max_) {
      max_ = x;
    } else if (absX > max_) {
      auto t{max_ / absX}; // < 1.0
      auto tsq{t * t};
      sum_ *= tsq; // scale sum to reflect change to the max
      sum_ += tsq; // include a term for the previous max
      max_ = absX;
    } else { // absX <= max_
      auto t{absX / max_};
      sum_ += t * t;
    }
    return true;
  }
  template <typename A> bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(*array_.Element<A>(at));
  }

private:
  const Descriptor &array_;
  AccumType max_{0}; // value (m) with largest magnitude
  AccumType sum_{0}; // sum((others(:)/m)**2)
};

template <int KIND> struct Norm2Helper {
  void operator()(Descriptor &result, const Descriptor &x, int dim,
      const Descriptor *mask, Terminator &terminator) const {
    DoMaxMinNorm2<TypeCategory::Real, KIND, Norm2Accumulator<KIND>>(
        result, x, dim, mask, "NORM2", terminator);
  }
};

extern "C" {
// TODO: REAL(2 & 3)
CppTypeFor<TypeCategory::Real, 4> RTNAME(Norm2_4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 4>(
      x, source, line, dim, mask, Norm2Accumulator<4>{x}, "NORM2");
}
CppTypeFor<TypeCategory::Real, 8> RTNAME(Norm2_8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 8>(
      x, source, line, dim, mask, Norm2Accumulator<8>{x}, "NORM2");
}
#if LONG_DOUBLE == 80
CppTypeFor<TypeCategory::Real, 10> RTNAME(Norm2_10)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 10>(
      x, source, line, dim, mask, Norm2Accumulator<10>{x}, "NORM2");
}
#elif LONG_DOUBLE == 128
CppTypeFor<TypeCategory::Real, 16> RTNAME(Norm2_16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Real, 16>(
      x, source, line, dim, mask, Norm2Accumulator<16>{x}, "NORM2");
}
#endif

void RTNAME(Norm2Dim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  Terminator terminator{source, line};
  auto type{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, type);
  if (type->first == TypeCategory::Real) {
    ApplyFloatingPointKind<Norm2Helper, void>(
        type->second, terminator, result, x, dim, mask, terminator);
  } else {
    terminator.Crash("NORM2: bad type code %d", x.type().raw());
  }
}
} // extern "C"
} // namespace Fortran::runtime
