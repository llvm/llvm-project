//===-- lib/runtime/tools.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang-rt/runtime/tools.h"
#include "flang-rt/runtime/terminator.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Fortran::runtime {

RT_OFFLOAD_API_GROUP_BEGIN

RT_API_ATTRS std::size_t TrimTrailingSpaces(const char *s, std::size_t n) {
  while (n > 0 && s[n - 1] == ' ') {
    --n;
  }
  return n;
}

RT_API_ATTRS OwningPtr<char> SaveDefaultCharacter(
    const char *s, std::size_t length, const Terminator &terminator) {
  if (s) {
    auto *p{static_cast<char *>(AllocateMemoryOrCrash(terminator, length + 1))};
    runtime::memcpy(p, s, length);
    p[length] = '\0';
    return OwningPtr<char>{p};
  } else {
    return OwningPtr<char>{};
  }
}

static RT_API_ATTRS bool CaseInsensitiveMatch(
    const char *value, std::size_t length, const char *possibility) {
  for (; length-- > 0; ++possibility) {
    char ch{*value++};
    if (ch >= 'a' && ch <= 'z') {
      ch += 'A' - 'a';
    }
    if (*possibility != ch) {
      if (*possibility != '\0' || ch != ' ') {
        return false;
      }
      // Ignore trailing blanks (12.5.6.2 p1)
      while (length-- > 0) {
        if (*value++ != ' ') {
          return false;
        }
      }
      return true;
    }
  }
  return *possibility == '\0';
}

RT_API_ATTRS int IdentifyValue(
    const char *value, std::size_t length, const char *possibilities[]) {
  if (value) {
    for (int j{0}; possibilities[j]; ++j) {
      if (CaseInsensitiveMatch(value, length, possibilities[j])) {
        return j;
      }
    }
  }
  return -1;
}

RT_API_ATTRS void ToFortranDefaultCharacter(
    char *to, std::size_t toLength, const char *from) {
  std::size_t len{Fortran::runtime::strlen(from)};
  if (len < toLength) {
    runtime::memcpy(to, from, len);
    runtime::memset(to + len, ' ', toLength - len);
  } else {
    runtime::memcpy(to, from, toLength);
  }
}

RT_API_ATTRS void CheckConformability(const Descriptor &to, const Descriptor &x,
    Terminator &terminator, const char *funcName, const char *toName,
    const char *xName) {
  if (x.rank() == 0) {
    return; // scalar conforms with anything
  }
  int rank{to.rank()};
  if (x.rank() != rank) {
    terminator.Crash(
        "Incompatible array arguments to %s: %s has rank %d but %s has rank %d",
        funcName, toName, rank, xName, x.rank());
  } else {
    for (int j{0}; j < rank; ++j) {
      auto toExtent{static_cast<std::int64_t>(to.GetDimension(j).Extent())};
      auto xExtent{static_cast<std::int64_t>(x.GetDimension(j).Extent())};
      if (xExtent != toExtent) {
        terminator.Crash("Incompatible array arguments to %s: dimension %d of "
                         "%s has extent %" PRId64 " but %s has extent %" PRId64,
            funcName, j + 1, toName, toExtent, xName, xExtent);
      }
    }
  }
}

RT_API_ATTRS void CheckIntegerKind(
    Terminator &terminator, int kind, const char *intrinsic) {
  if (kind < 1 || kind > 16 || (kind & (kind - 1)) != 0) {
    terminator.Crash("not yet implemented: INTEGER(KIND=%d) in %s intrinsic",
        intrinsic, kind);
  }
}

template <typename P, int RANK>
RT_API_ATTRS void ShallowCopyDiscontiguousToDiscontiguous(
    const Descriptor &to, const Descriptor &from) {
  DescriptorIterator<RANK> toIt{to};
  DescriptorIterator<RANK> fromIt{from};
  // Knowing the size at compile time can enable memcpy inlining optimisations
  constexpr std::size_t typeElementBytes{sizeof(P)};
  // We might still need to check the actual size as a fallback
  std::size_t elementBytes{to.ElementBytes()};
  for (std::size_t n{to.Elements()}; n-- > 0;
      toIt.Advance(), fromIt.Advance()) {
    // typeElementBytes == 1 when P is a char - the non-specialised case
    if constexpr (typeElementBytes != 1) {
      runtime::memcpy(
          toIt.template Get<P>(), fromIt.template Get<P>(), typeElementBytes);
    } else {
      runtime::memcpy(
          toIt.template Get<P>(), fromIt.template Get<P>(), elementBytes);
    }
  }
}

// Explicitly instantiate the default case to conform to the C++ standard
template RT_API_ATTRS void ShallowCopyDiscontiguousToDiscontiguous<char, -1>(
    const Descriptor &to, const Descriptor &from);

template <typename P, int RANK>
RT_API_ATTRS void ShallowCopyDiscontiguousToContiguous(
    const Descriptor &to, const Descriptor &from) {
  char *toAt{to.OffsetElement()};
  constexpr std::size_t typeElementBytes{sizeof(P)};
  std::size_t elementBytes{to.ElementBytes()};
  DescriptorIterator<RANK> fromIt{from};
  for (std::size_t n{to.Elements()}; n-- > 0;
      toAt += elementBytes, fromIt.Advance()) {
    if constexpr (typeElementBytes != 1) {
      runtime::memcpy(toAt, fromIt.template Get<P>(), typeElementBytes);
    } else {
      runtime::memcpy(toAt, fromIt.template Get<P>(), elementBytes);
    }
  }
}

template RT_API_ATTRS void ShallowCopyDiscontiguousToContiguous<char, -1>(
    const Descriptor &to, const Descriptor &from);

template <typename P, int RANK>
RT_API_ATTRS void ShallowCopyContiguousToDiscontiguous(
    const Descriptor &to, const Descriptor &from) {
  char *fromAt{from.OffsetElement()};
  DescriptorIterator<RANK> toIt{to};
  constexpr std::size_t typeElementBytes{sizeof(P)};
  std::size_t elementBytes{to.ElementBytes()};
  for (std::size_t n{to.Elements()}; n-- > 0;
      toIt.Advance(), fromAt += elementBytes) {
    if constexpr (typeElementBytes != 1) {
      runtime::memcpy(toIt.template Get<P>(), fromAt, typeElementBytes);
    } else {
      runtime::memcpy(toIt.template Get<P>(), fromAt, elementBytes);
    }
  }
}

template RT_API_ATTRS void ShallowCopyContiguousToDiscontiguous<char, -1>(
    const Descriptor &to, const Descriptor &from);

// ShallowCopy helper for calling the correct specialised variant based on
// scenario
template <typename P, int RANK = -1>
RT_API_ATTRS void ShallowCopyInner(const Descriptor &to, const Descriptor &from,
    bool toIsContiguous, bool fromIsContiguous) {
  if (toIsContiguous) {
    if (fromIsContiguous) {
      runtime::memcpy(to.OffsetElement(), from.OffsetElement(),
          to.Elements() * to.ElementBytes());
    } else {
      ShallowCopyDiscontiguousToContiguous<P, RANK>(to, from);
    }
  } else {
    if (fromIsContiguous) {
      ShallowCopyContiguousToDiscontiguous<P, RANK>(to, from);
    } else {
      ShallowCopyDiscontiguousToDiscontiguous<P, RANK>(to, from);
    }
  }
}

// Most arrays are much closer to rank-1 than to maxRank.
// Doing the recursion upwards instead of downwards puts the more common
// cases earlier in the if-chain and has a tangible impact on performance.
template <typename P, int RANK> struct ShallowCopyRankSpecialize {
  static RT_API_ATTRS bool execute(const Descriptor &to, const Descriptor &from,
      bool toIsContiguous, bool fromIsContiguous) {
    if (to.rank() == RANK && from.rank() == RANK) {
      ShallowCopyInner<P, RANK>(to, from, toIsContiguous, fromIsContiguous);
      return true;
    }
    return ShallowCopyRankSpecialize<P, RANK + 1>::execute(
        to, from, toIsContiguous, fromIsContiguous);
  }
};

template <typename P> struct ShallowCopyRankSpecialize<P, maxRank + 1> {
  static RT_API_ATTRS bool execute(const Descriptor &to, const Descriptor &from,
      bool toIsContiguous, bool fromIsContiguous) {
    return false;
  }
};

// ShallowCopy helper for specialising the variants based on array rank
template <typename P>
RT_API_ATTRS void ShallowCopyRank(const Descriptor &to, const Descriptor &from,
    bool toIsContiguous, bool fromIsContiguous) {
  // Try to call a specialised ShallowCopy variant from rank-1 up to maxRank
  bool specialized{ShallowCopyRankSpecialize<P, 1>::execute(
      to, from, toIsContiguous, fromIsContiguous)};
  if (!specialized) {
    ShallowCopyInner<P>(to, from, toIsContiguous, fromIsContiguous);
  }
}

RT_API_ATTRS void ShallowCopy(const Descriptor &to, const Descriptor &from,
    bool toIsContiguous, bool fromIsContiguous) {
  std::size_t elementBytes{to.ElementBytes()};
  // Checking the type at runtime and making sure the pointer passed to memcpy
  // has a type that matches the element type makes it possible for the compiler
  // to optimise out the memcpy calls altogether and can substantially improve
  // performance for some applications.
  if (to.type().IsInteger()) {
    if (elementBytes == sizeof(int64_t)) {
      ShallowCopyRank<int64_t>(to, from, toIsContiguous, fromIsContiguous);
    } else if (elementBytes == sizeof(int32_t)) {
      ShallowCopyRank<int32_t>(to, from, toIsContiguous, fromIsContiguous);
    } else if (elementBytes == sizeof(int16_t)) {
      ShallowCopyRank<int16_t>(to, from, toIsContiguous, fromIsContiguous);
#if defined USING_NATIVE_INT128_T
    } else if (elementBytes == sizeof(__int128_t)) {
      ShallowCopyRank<__int128_t>(to, from, toIsContiguous, fromIsContiguous);
#endif
    } else {
      ShallowCopyRank<char>(to, from, toIsContiguous, fromIsContiguous);
    }
  } else if (to.type().IsReal()) {
    if (elementBytes == sizeof(double)) {
      ShallowCopyRank<double>(to, from, toIsContiguous, fromIsContiguous);
    } else if (elementBytes == sizeof(float)) {
      ShallowCopyRank<float>(to, from, toIsContiguous, fromIsContiguous);
    } else {
      ShallowCopyRank<char>(to, from, toIsContiguous, fromIsContiguous);
    }
  } else {
    ShallowCopyRank<char>(to, from, toIsContiguous, fromIsContiguous);
  }
}

RT_API_ATTRS void ShallowCopy(const Descriptor &to, const Descriptor &from) {
  ShallowCopy(to, from, to.IsContiguous(), from.IsContiguous());
}

RT_API_ATTRS char *EnsureNullTerminated(
    char *str, std::size_t length, Terminator &terminator) {
  if (runtime::memchr(str, '\0', length) == nullptr) {
    char *newCmd{(char *)AllocateMemoryOrCrash(terminator, length + 1)};
    runtime::memcpy(newCmd, str, length);
    newCmd[length] = '\0';
    return newCmd;
  } else {
    return str;
  }
}

RT_API_ATTRS bool IsValidCharDescriptor(const Descriptor *value) {
  return value && value->IsAllocated() &&
      value->type() == TypeCode(TypeCategory::Character, 1) &&
      value->rank() == 0;
}

RT_API_ATTRS bool IsValidIntDescriptor(const Descriptor *intVal) {
  // Check that our descriptor is allocated and is a scalar integer with
  // kind != 1 (i.e. with a large enough decimal exponent range).
  return intVal && intVal->IsAllocated() && intVal->rank() == 0 &&
      intVal->type().IsInteger() && intVal->type().GetCategoryAndKind() &&
      intVal->type().GetCategoryAndKind()->second != 1;
}

RT_API_ATTRS std::int32_t CopyCharsToDescriptor(const Descriptor &value,
    const char *rawValue, std::size_t rawValueLength, const Descriptor *errmsg,
    std::size_t offset) {

  const std::int64_t toCopy{std::min(static_cast<std::int64_t>(rawValueLength),
      static_cast<std::int64_t>(value.ElementBytes() - offset))};
  if (toCopy < 0) {
    return ToErrmsg(errmsg, StatValueTooShort);
  }

  runtime::memcpy(value.OffsetElement(offset), rawValue, toCopy);

  if (static_cast<std::int64_t>(rawValueLength) > toCopy) {
    return ToErrmsg(errmsg, StatValueTooShort);
  }

  return StatOk;
}

RT_API_ATTRS void StoreIntToDescriptor(
    const Descriptor *length, std::int64_t value, Terminator &terminator) {
  auto typeCode{length->type().GetCategoryAndKind()};
  int kind{typeCode->second};
  ApplyIntegerKind<StoreIntegerAt, void>(
      kind, terminator, *length, /* atIndex = */ 0, value);
}

template <int KIND> struct FitsInIntegerKind {
  RT_API_ATTRS bool operator()([[maybe_unused]] std::int64_t value) {
    if constexpr (KIND >= 8) {
      return true;
    } else {
      return value <=
          std::numeric_limits<
              CppTypeFor<Fortran::common::TypeCategory::Integer, KIND>>::max();
    }
  }
};

// Utility: establishes & allocates the result array for a partial
// reduction (i.e., one with DIM=).
RT_API_ATTRS void CreatePartialReductionResult(Descriptor &result,
    const Descriptor &x, std::size_t resultElementSize, int dim,
    Terminator &terminator, const char *intrinsic, TypeCode typeCode) {
  int xRank{x.rank()};
  if (dim < 1 || dim > xRank) {
    terminator.Crash(
        "%s: bad DIM=%d for ARRAY with rank %d", intrinsic, dim, xRank);
  }
  int zeroBasedDim{dim - 1};
  SubscriptValue resultExtent[maxRank];
  for (int j{0}; j < zeroBasedDim; ++j) {
    resultExtent[j] = x.GetDimension(j).Extent();
  }
  for (int j{zeroBasedDim + 1}; j < xRank; ++j) {
    resultExtent[j - 1] = x.GetDimension(j).Extent();
  }
  result.Establish(typeCode, resultElementSize, nullptr, xRank - 1,
      resultExtent, CFI_attribute_allocatable);
  for (int j{0}; j + 1 < xRank; ++j) {
    result.GetDimension(j).SetBounds(1, resultExtent[j]);
  }
  if (int stat{result.Allocate(kNoAsyncObject)}) {
    terminator.Crash(
        "%s: could not allocate memory for result; STAT=%d", intrinsic, stat);
  }
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
