//===-- include/flang-rt/runtime/tools.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_RT_RUNTIME_TOOLS_H_
#define FLANG_RT_RUNTIME_TOOLS_H_

#include "descriptor.h"
#include "memory.h"
#include "stat.h"
#include "terminator.h"
#include "flang/Common/optional.h"
#include "flang/Runtime/cpp-type.h"
#include "flang/Runtime/freestanding-tools.h"
#include <cstring>
#include <functional>
#include <map>
#include <type_traits>

/// \macro RT_PRETTY_FUNCTION
/// Gets a user-friendly looking function signature for the current scope
/// using the best available method on each platform.  The exact format of the
/// resulting string is implementation specific and non-portable, so this should
/// only be used, for example, for logging or diagnostics.
/// Copy of LLVM_PRETTY_FUNCTION
#if defined(_MSC_VER)
#define RT_PRETTY_FUNCTION __FUNCSIG__
#elif defined(__GNUC__) || defined(__clang__)
#define RT_PRETTY_FUNCTION __PRETTY_FUNCTION__
#else
#define RT_PRETTY_FUNCTION __func__
#endif

#if defined(RT_DEVICE_COMPILATION)
// Use the pseudo lock and pseudo file unit implementations
// for the device.
#define RT_USE_PSEUDO_LOCK 1
#define RT_USE_PSEUDO_FILE_UNIT 1
#endif

namespace Fortran::runtime {

class Terminator;

RT_API_ATTRS std::size_t TrimTrailingSpaces(const char *, std::size_t);

RT_API_ATTRS OwningPtr<char> SaveDefaultCharacter(
    const char *, std::size_t, const Terminator &);

// For validating and recognizing default CHARACTER values in a
// case-insensitive manner.  Returns the zero-based index into the
// null-terminated array of upper-case possibilities when the value is valid,
// or -1 when it has no match.
RT_API_ATTRS int IdentifyValue(
    const char *value, std::size_t length, const char *possibilities[]);

// Truncates or pads as necessary
RT_API_ATTRS void ToFortranDefaultCharacter(
    char *to, std::size_t toLength, const char *from);

// Utilities for dealing with elemental LOGICAL arguments
inline RT_API_ATTRS bool IsLogicalElementTrue(
    const Descriptor &logical, const SubscriptValue at[]) {
  // A LOGICAL value is false if and only if all of its bytes are zero.
  const char *p{logical.Element<char>(at)};
  for (std::size_t j{logical.ElementBytes()}; j-- > 0; ++p) {
    if (*p) {
      return true;
    }
  }
  return false;
}
inline RT_API_ATTRS bool IsLogicalScalarTrue(const Descriptor &logical) {
  // A LOGICAL value is false if and only if all of its bytes are zero.
  const char *p{logical.OffsetElement<char>()};
  for (std::size_t j{logical.ElementBytes()}; j-- > 0; ++p) {
    if (*p) {
      return true;
    }
  }
  return false;
}

// Check array conformability; a scalar 'x' conforms.  Crashes on error.
RT_API_ATTRS void CheckConformability(const Descriptor &to, const Descriptor &x,
    Terminator &, const char *funcName, const char *toName,
    const char *fromName);

// Helper to store integer value in result[at].
template <int KIND> struct StoreIntegerAt {
  RT_API_ATTRS void operator()(const Fortran::runtime::Descriptor &result,
      std::size_t at, std::int64_t value) const {
    *result.ZeroBasedIndexedElement<Fortran::runtime::CppTypeFor<
        Fortran::common::TypeCategory::Integer, KIND>>(at) = value;
  }
};

// Helper to store floating value in result[at].
template <int KIND> struct StoreFloatingPointAt {
  RT_API_ATTRS void operator()(const Fortran::runtime::Descriptor &result,
      std::size_t at, std::double_t value) const {
    *result.ZeroBasedIndexedElement<Fortran::runtime::CppTypeFor<
        Fortran::common::TypeCategory::Real, KIND>>(at) = value;
  }
};

// Validate a KIND= argument
RT_API_ATTRS void CheckIntegerKind(
    Terminator &, int kind, const char *intrinsic);

template <typename TO, typename FROM>
inline RT_API_ATTRS void PutContiguousConverted(
    TO *to, FROM *from, std::size_t count) {
  while (count-- > 0) {
    *to++ = *from++;
  }
}

static inline RT_API_ATTRS std::int64_t GetInt64(
    const char *p, std::size_t bytes, Terminator &terminator) {
  switch (bytes) {
  case 1:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 1> *>(p);
  case 2:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 2> *>(p);
  case 4:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 4> *>(p);
  case 8:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 8> *>(p);
  default:
    terminator.Crash("GetInt64: no case for %zd bytes", bytes);
  }
}

static inline RT_API_ATTRS Fortran::common::optional<std::int64_t> GetInt64Safe(
    const char *p, std::size_t bytes, Terminator &terminator) {
  switch (bytes) {
  case 1:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 1> *>(p);
  case 2:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 2> *>(p);
  case 4:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 4> *>(p);
  case 8:
    return *reinterpret_cast<const CppTypeFor<TypeCategory::Integer, 8> *>(p);
  case 16: {
    using Int128 = CppTypeFor<TypeCategory::Integer, 16>;
    auto n{*reinterpret_cast<const Int128 *>(p)};
    std::int64_t result{static_cast<std::int64_t>(n)};
    if (static_cast<Int128>(result) == n) {
      return result;
    }
    return Fortran::common::nullopt;
  }
  default:
    terminator.Crash("GetInt64Safe: no case for %zd bytes", bytes);
  }
}

template <typename INT>
inline RT_API_ATTRS bool SetInteger(INT &x, int kind, std::int64_t value) {
  switch (kind) {
  case 1:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 1> &>(x) = value;
    return value == reinterpret_cast<CppTypeFor<TypeCategory::Integer, 1> &>(x);
  case 2:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 2> &>(x) = value;
    return value == reinterpret_cast<CppTypeFor<TypeCategory::Integer, 2> &>(x);
  case 4:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 4> &>(x) = value;
    return value == reinterpret_cast<CppTypeFor<TypeCategory::Integer, 4> &>(x);
  case 8:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 8> &>(x) = value;
    return value == reinterpret_cast<CppTypeFor<TypeCategory::Integer, 8> &>(x);
  default:
    return false;
  }
}

// Maps intrinsic runtime type category and kind values to the appropriate
// instantiation of a function object template and calls it with the supplied
// arguments.
template <template <TypeCategory, int> class FUNC, typename RESULT,
    typename... A>
inline RT_API_ATTRS RESULT ApplyType(
    TypeCategory cat, int kind, Terminator &terminator, A &&...x) {
  switch (cat) {
  case TypeCategory::Integer:
    switch (kind) {
    case 1:
      return FUNC<TypeCategory::Integer, 1>{}(std::forward<A>(x)...);
    case 2:
      return FUNC<TypeCategory::Integer, 2>{}(std::forward<A>(x)...);
    case 4:
      return FUNC<TypeCategory::Integer, 4>{}(std::forward<A>(x)...);
    case 8:
      return FUNC<TypeCategory::Integer, 8>{}(std::forward<A>(x)...);
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
    case 16:
      return FUNC<TypeCategory::Integer, 16>{}(std::forward<A>(x)...);
#endif
    default:
      terminator.Crash("not yet implemented: INTEGER(KIND=%d)", kind);
    }
  case TypeCategory::Unsigned:
    switch (kind) {
    case 1:
      return FUNC<TypeCategory::Unsigned, 1>{}(std::forward<A>(x)...);
    case 2:
      return FUNC<TypeCategory::Unsigned, 2>{}(std::forward<A>(x)...);
    case 4:
      return FUNC<TypeCategory::Unsigned, 4>{}(std::forward<A>(x)...);
    case 8:
      return FUNC<TypeCategory::Unsigned, 8>{}(std::forward<A>(x)...);
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
    case 16:
      return FUNC<TypeCategory::Unsigned, 16>{}(std::forward<A>(x)...);
#endif
    default:
      terminator.Crash("not yet implemented: UNSIGNED(KIND=%d)", kind);
    }
  case TypeCategory::Real:
    switch (kind) {
#if 0 // TODO: REAL(2 & 3)
    case 2:
      return FUNC<TypeCategory::Real, 2>{}(std::forward<A>(x)...);
    case 3:
      return FUNC<TypeCategory::Real, 3>{}(std::forward<A>(x)...);
#endif
    case 4:
      return FUNC<TypeCategory::Real, 4>{}(std::forward<A>(x)...);
    case 8:
      return FUNC<TypeCategory::Real, 8>{}(std::forward<A>(x)...);
    case 10:
      if constexpr (HasCppTypeFor<TypeCategory::Real, 10>) {
        return FUNC<TypeCategory::Real, 10>{}(std::forward<A>(x)...);
      }
      break;
    case 16:
      if constexpr (HasCppTypeFor<TypeCategory::Real, 16>) {
        return FUNC<TypeCategory::Real, 16>{}(std::forward<A>(x)...);
      }
      break;
    }
    terminator.Crash("not yet implemented: REAL(KIND=%d)", kind);
  case TypeCategory::Complex:
    switch (kind) {
#if 0 // TODO: COMPLEX(2 & 3)
    case 2:
      return FUNC<TypeCategory::Complex, 2>{}(std::forward<A>(x)...);
    case 3:
      return FUNC<TypeCategory::Complex, 3>{}(std::forward<A>(x)...);
#endif
    case 4:
      return FUNC<TypeCategory::Complex, 4>{}(std::forward<A>(x)...);
    case 8:
      return FUNC<TypeCategory::Complex, 8>{}(std::forward<A>(x)...);
    case 10:
      if constexpr (HasCppTypeFor<TypeCategory::Real, 10>) {
        return FUNC<TypeCategory::Complex, 10>{}(std::forward<A>(x)...);
      }
      break;
    case 16:
      if constexpr (HasCppTypeFor<TypeCategory::Real, 16>) {
        return FUNC<TypeCategory::Complex, 16>{}(std::forward<A>(x)...);
      }
      break;
    }
    terminator.Crash("not yet implemented: COMPLEX(KIND=%d)", kind);
  case TypeCategory::Character:
    switch (kind) {
    case 1:
      return FUNC<TypeCategory::Character, 1>{}(std::forward<A>(x)...);
    case 2:
      return FUNC<TypeCategory::Character, 2>{}(std::forward<A>(x)...);
    case 4:
      return FUNC<TypeCategory::Character, 4>{}(std::forward<A>(x)...);
    default:
      terminator.Crash("not yet implemented: CHARACTER(KIND=%d)", kind);
    }
  case TypeCategory::Logical:
    switch (kind) {
    case 1:
      return FUNC<TypeCategory::Logical, 1>{}(std::forward<A>(x)...);
    case 2:
      return FUNC<TypeCategory::Logical, 2>{}(std::forward<A>(x)...);
    case 4:
      return FUNC<TypeCategory::Logical, 4>{}(std::forward<A>(x)...);
    case 8:
      return FUNC<TypeCategory::Logical, 8>{}(std::forward<A>(x)...);
    default:
      terminator.Crash("not yet implemented: LOGICAL(KIND=%d)", kind);
    }
  default:
    terminator.Crash(
        "not yet implemented: type category(%d)", static_cast<int>(cat));
  }
}

// Maps a runtime INTEGER kind value to the appropriate instantiation of
// a function object template and calls it with the supplied arguments.
template <template <int KIND> class FUNC, typename RESULT, typename... A>
inline RT_API_ATTRS RESULT ApplyIntegerKind(
    int kind, Terminator &terminator, A &&...x) {
  switch (kind) {
  case 1:
    return FUNC<1>{}(std::forward<A>(x)...);
  case 2:
    return FUNC<2>{}(std::forward<A>(x)...);
  case 4:
    return FUNC<4>{}(std::forward<A>(x)...);
  case 8:
    return FUNC<8>{}(std::forward<A>(x)...);
#if defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T
  case 16:
    return FUNC<16>{}(std::forward<A>(x)...);
#endif
  default:
    terminator.Crash("not yet implemented: INTEGER/UNSIGNED(KIND=%d)", kind);
  }
}

template <template <int KIND> class FUNC, typename RESULT,
    bool NEEDSMATH = false, typename... A>
inline RT_API_ATTRS RESULT ApplyFloatingPointKind(
    int kind, Terminator &terminator, A &&...x) {
  switch (kind) {
#if 0 // TODO: REAL/COMPLEX (2 & 3)
  case 2:
    return FUNC<2>{}(std::forward<A>(x)...);
  case 3:
    return FUNC<3>{}(std::forward<A>(x)...);
#endif
  case 4:
    return FUNC<4>{}(std::forward<A>(x)...);
  case 8:
    return FUNC<8>{}(std::forward<A>(x)...);
  case 10:
    if constexpr (HasCppTypeFor<TypeCategory::Real, 10>) {
      return FUNC<10>{}(std::forward<A>(x)...);
    }
    break;
  case 16:
    if constexpr (HasCppTypeFor<TypeCategory::Real, 16>) {
      // If FUNC implemenation relies on FP math functions,
      // then we should not be here. The compiler should have
      // generated a call to an entry in the libflang_rt.quadmath
      // library.
      if constexpr (!NEEDSMATH) {
        return FUNC<16>{}(std::forward<A>(x)...);
      }
    }
    break;
  }
  terminator.Crash("not yet implemented: REAL/COMPLEX(KIND=%d)", kind);
}

template <template <int KIND> class FUNC, typename RESULT, typename... A>
inline RT_API_ATTRS RESULT ApplyCharacterKind(
    int kind, Terminator &terminator, A &&...x) {
  switch (kind) {
  case 1:
    return FUNC<1>{}(std::forward<A>(x)...);
  case 2:
    return FUNC<2>{}(std::forward<A>(x)...);
  case 4:
    return FUNC<4>{}(std::forward<A>(x)...);
  default:
    terminator.Crash("not yet implemented: CHARACTER(KIND=%d)", kind);
  }
}

template <template <int KIND> class FUNC, typename RESULT, typename... A>
inline RT_API_ATTRS RESULT ApplyLogicalKind(
    int kind, Terminator &terminator, A &&...x) {
  switch (kind) {
  case 1:
    return FUNC<1>{}(std::forward<A>(x)...);
  case 2:
    return FUNC<2>{}(std::forward<A>(x)...);
  case 4:
    return FUNC<4>{}(std::forward<A>(x)...);
  case 8:
    return FUNC<8>{}(std::forward<A>(x)...);
  default:
    terminator.Crash("not yet implemented: LOGICAL(KIND=%d)", kind);
  }
}

// Calculate result type of (X op Y) for *, //, DOT_PRODUCT, &c.
Fortran::common::optional<
    std::pair<TypeCategory, int>> inline constexpr RT_API_ATTRS
GetResultType(TypeCategory xCat, int xKind, TypeCategory yCat, int yKind) {
  int maxKind{std::max(xKind, yKind)};
  switch (xCat) {
  case TypeCategory::Integer:
    switch (yCat) {
    case TypeCategory::Integer:
      return std::make_pair(TypeCategory::Integer, maxKind);
    case TypeCategory::Real:
    case TypeCategory::Complex:
#if !(defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T)
      if (xKind == 16) {
        break;
      }
#endif
      return std::make_pair(yCat, yKind);
    default:
      break;
    }
    break;
  case TypeCategory::Unsigned:
    switch (yCat) {
    case TypeCategory::Unsigned:
      return std::make_pair(TypeCategory::Unsigned, maxKind);
    case TypeCategory::Real:
    case TypeCategory::Complex:
#if !(defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T)
      if (xKind == 16) {
        break;
      }
#endif
      return std::make_pair(yCat, yKind);
    default:
      break;
    }
    break;
  case TypeCategory::Real:
    switch (yCat) {
    case TypeCategory::Integer:
    case TypeCategory::Unsigned:
#if !(defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T)
      if (yKind == 16) {
        break;
      }
#endif
      return std::make_pair(TypeCategory::Real, xKind);
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return std::make_pair(yCat, maxKind);
    default:
      break;
    }
    break;
  case TypeCategory::Complex:
    switch (yCat) {
    case TypeCategory::Integer:
    case TypeCategory::Unsigned:
#if !(defined __SIZEOF_INT128__ && !AVOID_NATIVE_UINT128_T)
      if (yKind == 16) {
        break;
      }
#endif
      return std::make_pair(TypeCategory::Complex, xKind);
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return std::make_pair(TypeCategory::Complex, maxKind);
    default:
      break;
    }
    break;
  case TypeCategory::Character:
    if (yCat == TypeCategory::Character) {
      return std::make_pair(TypeCategory::Character, maxKind);
    } else {
      return Fortran::common::nullopt;
    }
  case TypeCategory::Logical:
    if (yCat == TypeCategory::Logical) {
      return std::make_pair(TypeCategory::Logical, maxKind);
    } else {
      return Fortran::common::nullopt;
    }
  default:
    break;
  }
  return Fortran::common::nullopt;
}

// Accumulate floating-point results in (at least) double precision
template <TypeCategory CAT, int KIND>
using AccumulationType = CppTypeFor<CAT,
    CAT == TypeCategory::Real || CAT == TypeCategory::Complex
        ? std::max(KIND, static_cast<int>(sizeof(double)))
        : KIND>;

// memchr() for any character type
template <typename CHAR>
static inline RT_API_ATTRS const CHAR *FindCharacter(
    const CHAR *data, CHAR ch, std::size_t chars) {
  const CHAR *end{data + chars};
  for (const CHAR *p{data}; p < end; ++p) {
    if (*p == ch) {
      return p;
    }
  }
  return nullptr;
}

template <>
inline RT_API_ATTRS const char *FindCharacter(
    const char *data, char ch, std::size_t chars) {
  return reinterpret_cast<const char *>(
      runtime::memchr(data, static_cast<int>(ch), chars));
}

// Copy payload data from one allocated descriptor to another.
// Assumes element counts and element sizes match, and that both
// descriptors are allocated.
template <typename P = char, int RANK = -1>
RT_API_ATTRS void ShallowCopyDiscontiguousToDiscontiguous(
    const Descriptor &to, const Descriptor &from);
template <typename P = char, int RANK = -1>
RT_API_ATTRS void ShallowCopyDiscontiguousToContiguous(
    const Descriptor &to, const Descriptor &from);
template <typename P = char, int RANK = -1>
RT_API_ATTRS void ShallowCopyContiguousToDiscontiguous(
    const Descriptor &to, const Descriptor &from);
RT_API_ATTRS void ShallowCopy(const Descriptor &to, const Descriptor &from,
    bool toIsContiguous, bool fromIsContiguous);
RT_API_ATTRS void ShallowCopy(const Descriptor &to, const Descriptor &from);

// Ensures that a character string is null-terminated, allocating a /p length +1
// size memory for null-terminator if necessary. Returns the original or a newly
// allocated null-terminated string (responsibility for deallocation is on the
// caller).
RT_API_ATTRS char *EnsureNullTerminated(
    char *str, std::size_t length, Terminator &terminator);

RT_API_ATTRS bool IsValidCharDescriptor(const Descriptor *value);

RT_API_ATTRS bool IsValidIntDescriptor(const Descriptor *intVal);

// Copy a null-terminated character array \p rawValue to descriptor \p value.
// The copy starts at the given \p offset, if not present then start at 0.
// If descriptor `errmsg` is provided, error messages will be stored to it.
// Returns stats specified in standard.
RT_API_ATTRS std::int32_t CopyCharsToDescriptor(const Descriptor &value,
    const char *rawValue, std::size_t rawValueLength,
    const Descriptor *errmsg = nullptr, std::size_t offset = 0);

RT_API_ATTRS void StoreIntToDescriptor(
    const Descriptor *length, std::int64_t value, Terminator &terminator);

// Defines a utility function for copying and padding characters
template <typename TO, typename FROM>
RT_API_ATTRS void CopyAndPad(
    TO *to, const FROM *from, std::size_t toChars, std::size_t fromChars) {
  if constexpr (sizeof(TO) != sizeof(FROM)) {
    std::size_t copyChars{std::min(toChars, fromChars)};
    for (std::size_t j{0}; j < copyChars; ++j) {
      to[j] = from[j];
    }
    for (std::size_t j{copyChars}; j < toChars; ++j) {
      to[j] = static_cast<TO>(' ');
    }
  } else if (toChars <= fromChars) {
    std::memcpy(to, from, toChars * sizeof(TO));
  } else {
    std::memcpy(to, from, std::min(toChars, fromChars) * sizeof(TO));
    for (std::size_t j{fromChars}; j < toChars; ++j) {
      to[j] = static_cast<TO>(' ');
    }
  }
}

RT_API_ATTRS void CreatePartialReductionResult(Descriptor &result,
    const Descriptor &x, std::size_t resultElementSize, int dim, Terminator &,
    const char *intrinsic, TypeCode);

} // namespace Fortran::runtime
#endif // FLANG_RT_RUNTIME_TOOLS_H_
