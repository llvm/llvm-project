//===-- runtime/reduction.cpp ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Implements ALL, ANY, COUNT, IALL, IANY, IPARITY, & PARITY for all required
// operand types and shapes.
//
// DOT_PRODUCT, FINDLOC, MATMUL, SUM, and PRODUCT are in their own eponymous
// source files.
// NORM2, MAXLOC, MINLOC, MAXVAL, and MINVAL are in extrema.cpp.

#include "flang/Runtime/reduction.h"
#include "reduction-templates.h"
#include "flang/Runtime/descriptor.h"
#include <cinttypes>

namespace Fortran::runtime {

// IALL, IANY, IPARITY

template <typename INTERMEDIATE> class IntegerAndAccumulator {
public:
  explicit RT_API_ATTRS IntegerAndAccumulator(const Descriptor &array)
      : array_{array} {}
  RT_API_ATTRS void Reinitialize() { and_ = ~INTERMEDIATE{0}; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(and_);
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    and_ &= *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE and_{~INTERMEDIATE{0}};
};

template <typename INTERMEDIATE> class IntegerOrAccumulator {
public:
  explicit RT_API_ATTRS IntegerOrAccumulator(const Descriptor &array)
      : array_{array} {}
  RT_API_ATTRS void Reinitialize() { or_ = 0; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(or_);
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    or_ |= *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE or_{0};
};

template <typename INTERMEDIATE> class IntegerXorAccumulator {
public:
  explicit RT_API_ATTRS IntegerXorAccumulator(const Descriptor &array)
      : array_{array} {}
  RT_API_ATTRS void Reinitialize() { xor_ = 0; }
  template <typename A>
  RT_API_ATTRS void GetResult(A *p, int /*zeroBasedDim*/ = -1) const {
    *p = static_cast<A>(xor_);
  }
  template <typename A>
  RT_API_ATTRS bool AccumulateAt(const SubscriptValue at[]) {
    xor_ ^= *array_.Element<A>(at);
    return true;
  }

private:
  const Descriptor &array_;
  INTERMEDIATE xor_{0};
};

extern "C" {
CppTypeFor<TypeCategory::Integer, 1> RTDEF(IAll1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      IntegerAndAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IALL",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(IAll2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      IntegerAndAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IALL",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(IAll4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      IntegerAndAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IALL",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(IAll8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      IntegerAndAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x}, "IALL",
      /*allowUnsignedForInteger=*/true);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(IAll16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask, IntegerAndAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "IALL", /*allowUnsignedForInteger=*/true);
}
#endif
void RTDEF(IAllDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  Terminator terminator{source, line};
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      catKind.has_value() &&
          (catKind->first == TypeCategory::Integer ||
              catKind->first == TypeCategory::Unsigned));
  PartialIntegerReduction<IntegerAndAccumulator>(
      result, x, dim, catKind->second, mask, "IALL", terminator);
}

CppTypeFor<TypeCategory::Integer, 1> RTDEF(IAny1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      IntegerOrAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IANY",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(IAny2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      IntegerOrAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IANY",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(IAny4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      IntegerOrAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IANY",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(IAny8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      IntegerOrAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x}, "IANY",
      /*allowUnsignedForInteger=*/true);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(IAny16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask, IntegerOrAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "IANY", /*allowUnsignedForInteger=*/true);
}
#endif
void RTDEF(IAnyDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  Terminator terminator{source, line};
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      catKind.has_value() &&
          (catKind->first == TypeCategory::Integer ||
              catKind->first == TypeCategory::Unsigned));
  PartialIntegerReduction<IntegerOrAccumulator>(
      result, x, dim, catKind->second, mask, "IANY", terminator);
}

CppTypeFor<TypeCategory::Integer, 1> RTDEF(IParity1)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 1>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IPARITY",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 2> RTDEF(IParity2)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 2>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IPARITY",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 4> RTDEF(IParity4)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 4>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 4>>{x}, "IPARITY",
      /*allowUnsignedForInteger=*/true);
}
CppTypeFor<TypeCategory::Integer, 8> RTDEF(IParity8)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 8>(x, source, line, dim, mask,
      IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 8>>{x}, "IPARITY",
      /*allowUnsignedForInteger=*/true);
}
#ifdef __SIZEOF_INT128__
CppTypeFor<TypeCategory::Integer, 16> RTDEF(IParity16)(const Descriptor &x,
    const char *source, int line, int dim, const Descriptor *mask) {
  return GetTotalReduction<TypeCategory::Integer, 16>(x, source, line, dim,
      mask, IntegerXorAccumulator<CppTypeFor<TypeCategory::Integer, 16>>{x},
      "IPARITY", /*allowUnsignedForInteger=*/true);
}
#endif
void RTDEF(IParityDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line, const Descriptor *mask) {
  Terminator terminator{source, line};
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator,
      catKind.has_value() &&
          (catKind->first == TypeCategory::Integer ||
              catKind->first == TypeCategory::Unsigned));
  PartialIntegerReduction<IntegerXorAccumulator>(
      result, x, dim, catKind->second, mask, "IPARITY", terminator);
}
}

// ALL, ANY, COUNT, & PARITY

enum class LogicalReduction { All, Any, Parity };

template <LogicalReduction REDUCTION> class LogicalAccumulator {
public:
  using Type = bool;
  explicit LogicalAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { result_ = REDUCTION == LogicalReduction::All; }
  bool Result() const { return result_; }
  bool Accumulate(bool x) {
    if constexpr (REDUCTION == LogicalReduction::Parity) {
      result_ = result_ != x;
    } else if (x != (REDUCTION == LogicalReduction::All)) {
      result_ = x;
      return false;
    }
    return true;
  }
  template <typename IGNORED = void>
  bool AccumulateAt(const SubscriptValue at[]) {
    return Accumulate(IsLogicalElementTrue(array_, at));
  }

private:
  const Descriptor &array_;
  bool result_{REDUCTION == LogicalReduction::All};
};

template <typename ACCUMULATOR>
inline auto GetTotalLogicalReduction(const Descriptor &x, const char *source,
    int line, int dim, ACCUMULATOR &&accumulator, const char *intrinsic) ->
    typename ACCUMULATOR::Type {
  Terminator terminator{source, line};
  if (dim < 0 || dim > 1) {
    terminator.Crash("%s: bad DIM=%d for ARRAY with rank=1", intrinsic, dim);
  }
  SubscriptValue xAt[maxRank];
  x.GetLowerBounds(xAt);
  for (auto elements{x.Elements()}; elements--; x.IncrementSubscripts(xAt)) {
    if (!accumulator.AccumulateAt(xAt)) {
      break; // cut short, result is known
    }
  }
  return accumulator.Result();
}

template <typename ACCUMULATOR>
inline auto ReduceLogicalDimToScalar(const Descriptor &x, int zeroBasedDim,
    SubscriptValue subscripts[]) -> typename ACCUMULATOR::Type {
  ACCUMULATOR accumulator{x};
  SubscriptValue xAt[maxRank];
  GetExpandedSubscripts(xAt, x, zeroBasedDim, subscripts);
  const auto &dim{x.GetDimension(zeroBasedDim)};
  SubscriptValue at{dim.LowerBound()};
  for (auto n{dim.Extent()}; n-- > 0; ++at) {
    xAt[zeroBasedDim] = at;
    if (!accumulator.AccumulateAt(xAt)) {
      break;
    }
  }
  return accumulator.Result();
}

template <LogicalReduction REDUCTION> struct LogicalReduceHelper {
  template <int KIND> struct Functor {
    void operator()(Descriptor &result, const Descriptor &x, int dim,
        Terminator &terminator, const char *intrinsic) const {
      // Standard requires result to have same LOGICAL kind as argument.
      CreatePartialReductionResult(
          result, x, x.ElementBytes(), dim, terminator, intrinsic, x.type());
      SubscriptValue at[maxRank];
      result.GetLowerBounds(at);
      INTERNAL_CHECK(result.rank() == 0 || at[0] == 1);
      using CppType = CppTypeFor<TypeCategory::Logical, KIND>;
      for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
        *result.Element<CppType>(at) =
            ReduceLogicalDimToScalar<LogicalAccumulator<REDUCTION>>(
                x, dim - 1, at);
      }
    }
  };
};

template <LogicalReduction REDUCTION>
inline void DoReduceLogicalDimension(Descriptor &result, const Descriptor &x,
    int dim, Terminator &terminator, const char *intrinsic) {
  auto catKind{x.type().GetCategoryAndKind()};
  RUNTIME_CHECK(terminator, catKind && catKind->first == TypeCategory::Logical);
  ApplyLogicalKind<LogicalReduceHelper<REDUCTION>::template Functor, void>(
      catKind->second, terminator, result, x, dim, terminator, intrinsic);
}

// COUNT

class CountAccumulator {
public:
  using Type = std::int64_t;
  explicit CountAccumulator(const Descriptor &array) : array_{array} {}
  void Reinitialize() { result_ = 0; }
  Type Result() const { return result_; }
  template <typename IGNORED = void>
  bool AccumulateAt(const SubscriptValue at[]) {
    if (IsLogicalElementTrue(array_, at)) {
      ++result_;
    }
    return true;
  }

private:
  const Descriptor &array_;
  Type result_{0};
};

template <int KIND> struct CountDimension {
  void operator()(Descriptor &result, const Descriptor &x, int dim,
      Terminator &terminator) const {
    // Element size of the descriptor descriptor is the size
    // of {TypeCategory::Integer, KIND}.
    CreatePartialReductionResult(result, x,
        Descriptor::BytesFor(TypeCategory::Integer, KIND), dim, terminator,
        "COUNT", TypeCode{TypeCategory::Integer, KIND});
    SubscriptValue at[maxRank];
    result.GetLowerBounds(at);
    INTERNAL_CHECK(result.rank() == 0 || at[0] == 1);
    using CppType = CppTypeFor<TypeCategory::Integer, KIND>;
    for (auto n{result.Elements()}; n-- > 0; result.IncrementSubscripts(at)) {
      *result.Element<CppType>(at) =
          ReduceLogicalDimToScalar<CountAccumulator>(x, dim - 1, at);
    }
  }
};

extern "C" {
RT_EXT_API_GROUP_BEGIN

bool RTDEF(All)(const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(x, source, line, dim,
      LogicalAccumulator<LogicalReduction::All>{x}, "ALL");
}
void RTDEF(AllDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<LogicalReduction::All>(
      result, x, dim, terminator, "ALL");
}

bool RTDEF(Any)(const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(x, source, line, dim,
      LogicalAccumulator<LogicalReduction::Any>{x}, "ANY");
}
void RTDEF(AnyDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<LogicalReduction::Any>(
      result, x, dim, terminator, "ANY");
}

std::int64_t RTDEF(Count)(
    const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(
      x, source, line, dim, CountAccumulator{x}, "COUNT");
}

void RTDEF(CountDim)(Descriptor &result, const Descriptor &x, int dim, int kind,
    const char *source, int line) {
  Terminator terminator{source, line};
  ApplyIntegerKind<CountDimension, void>(
      kind, terminator, result, x, dim, terminator);
}

bool RTDEF(Parity)(const Descriptor &x, const char *source, int line, int dim) {
  return GetTotalLogicalReduction(x, source, line, dim,
      LogicalAccumulator<LogicalReduction::Parity>{x}, "PARITY");
}
void RTDEF(ParityDim)(Descriptor &result, const Descriptor &x, int dim,
    const char *source, int line) {
  Terminator terminator{source, line};
  DoReduceLogicalDimension<LogicalReduction::Parity>(
      result, x, dim, terminator, "PARITY");
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
