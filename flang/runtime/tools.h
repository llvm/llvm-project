//===-- runtime/tools.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_RUNTIME_TOOLS_H_
#define FORTRAN_RUNTIME_TOOLS_H_

#include "cpp-type.h"
#include "descriptor.h"
#include "memory.h"
#include "terminator.h"
#include "flang/Common/long-double.h"
#include <functional>
#include <map>
#include <type_traits>

namespace Fortran::runtime {

class Terminator;

std::size_t TrimTrailingSpaces(const char *, std::size_t);

OwningPtr<char> SaveDefaultCharacter(
    const char *, std::size_t, const Terminator &);

// For validating and recognizing default CHARACTER values in a
// case-insensitive manner.  Returns the zero-based index into the
// null-terminated array of upper-case possibilities when the value is valid,
// or -1 when it has no match.
int IdentifyValue(
    const char *value, std::size_t length, const char *possibilities[]);

// Truncates or pads as necessary
void ToFortranDefaultCharacter(
    char *to, std::size_t toLength, const char *from);

// Utility for dealing with elemental LOGICAL arguments
inline bool IsLogicalElementTrue(
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

// Check array conformability; a scalar 'x' conforms.  Crashes on error.
void CheckConformability(const Descriptor &to, const Descriptor &x,
    Terminator &, const char *funcName, const char *toName,
    const char *fromName);

// Validate a KIND= argument
void CheckIntegerKind(Terminator &, int kind, const char *intrinsic);

template <typename TO, typename FROM>
inline void PutContiguousConverted(TO *to, FROM *from, std::size_t count) {
  while (count-- > 0) {
    *to++ = *from++;
  }
}

static inline std::int64_t GetInt64(
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

template <typename INT>
inline bool SetInteger(INT &x, int kind, std::int64_t value) {
  switch (kind) {
  case 1:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 1> &>(x) = value;
    return true;
  case 2:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 2> &>(x) = value;
    return true;
  case 4:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 4> &>(x) = value;
    return true;
  case 8:
    reinterpret_cast<CppTypeFor<TypeCategory::Integer, 8> &>(x) = value;
    return true;
  default:
    return false;
  }
}

// Maps intrinsic runtime type category and kind values to the appropriate
// instantiation of a function object template and calls it with the supplied
// arguments.
template <template <TypeCategory, int> class FUNC, typename RESULT,
    typename... A>
inline RESULT ApplyType(
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
#ifdef __SIZEOF_INT128__
    case 16:
      return FUNC<TypeCategory::Integer, 16>{}(std::forward<A>(x)...);
#endif
    default:
      terminator.Crash("unsupported INTEGER(KIND=%d)", kind);
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
#if LONG_DOUBLE == 80
    case 10:
      return FUNC<TypeCategory::Real, 10>{}(std::forward<A>(x)...);
#elif LONG_DOUBLE == 128
    case 16:
      return FUNC<TypeCategory::Real, 16>{}(std::forward<A>(x)...);
#endif
    default:
      terminator.Crash("unsupported REAL(KIND=%d)", kind);
    }
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
#if LONG_DOUBLE == 80
    case 10:
      return FUNC<TypeCategory::Complex, 10>{}(std::forward<A>(x)...);
#elif LONG_DOUBLE == 128
    case 16:
      return FUNC<TypeCategory::Complex, 16>{}(std::forward<A>(x)...);
#endif
    default:
      terminator.Crash("unsupported COMPLEX(KIND=%d)", kind);
    }
  case TypeCategory::Character:
    switch (kind) {
    case 1:
      return FUNC<TypeCategory::Character, 1>{}(std::forward<A>(x)...);
    case 2:
      return FUNC<TypeCategory::Character, 2>{}(std::forward<A>(x)...);
    case 4:
      return FUNC<TypeCategory::Character, 4>{}(std::forward<A>(x)...);
    default:
      terminator.Crash("unsupported CHARACTER(KIND=%d)", kind);
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
      terminator.Crash("unsupported LOGICAL(KIND=%d)", kind);
    }
  default:
    terminator.Crash("unsupported type category(%d)", static_cast<int>(cat));
  }
}

// Maps a runtime INTEGER kind value to the appropriate instantiation of
// a function object template and calls it with the supplied arguments.
template <template <int KIND> class FUNC, typename RESULT, typename... A>
inline RESULT ApplyIntegerKind(int kind, Terminator &terminator, A &&...x) {
  switch (kind) {
  case 1:
    return FUNC<1>{}(std::forward<A>(x)...);
  case 2:
    return FUNC<2>{}(std::forward<A>(x)...);
  case 4:
    return FUNC<4>{}(std::forward<A>(x)...);
  case 8:
    return FUNC<8>{}(std::forward<A>(x)...);
#ifdef __SIZEOF_INT128__
  case 16:
    return FUNC<16>{}(std::forward<A>(x)...);
#endif
  default:
    terminator.Crash("unsupported INTEGER(KIND=%d)", kind);
  }
}

template <template <int KIND> class FUNC, typename RESULT, typename... A>
inline RESULT ApplyFloatingPointKind(
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
#if LONG_DOUBLE == 80
  case 10:
    return FUNC<10>{}(std::forward<A>(x)...);
#elif LONG_DOUBLE == 128
  case 16:
    return FUNC<16>{}(std::forward<A>(x)...);
#endif
  default:
    terminator.Crash("unsupported REAL/COMPLEX(KIND=%d)", kind);
  }
}

template <template <int KIND> class FUNC, typename RESULT, typename... A>
inline RESULT ApplyCharacterKind(int kind, Terminator &terminator, A &&...x) {
  switch (kind) {
  case 1:
    return FUNC<1>{}(std::forward<A>(x)...);
  case 2:
    return FUNC<2>{}(std::forward<A>(x)...);
  case 4:
    return FUNC<4>{}(std::forward<A>(x)...);
  default:
    terminator.Crash("unsupported CHARACTER(KIND=%d)", kind);
  }
}

template <template <int KIND> class FUNC, typename RESULT, typename... A>
inline RESULT ApplyLogicalKind(int kind, Terminator &terminator, A &&...x) {
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
    terminator.Crash("unsupported LOGICAL(KIND=%d)", kind);
  }
}

// Calculate result type of (X op Y) for *, //, DOT_PRODUCT, &c.
std::optional<std::pair<TypeCategory, int>> inline constexpr GetResultType(
    TypeCategory xCat, int xKind, TypeCategory yCat, int yKind) {
  int maxKind{std::max(xKind, yKind)};
  switch (xCat) {
  case TypeCategory::Integer:
    switch (yCat) {
    case TypeCategory::Integer:
      return std::make_pair(TypeCategory::Integer, maxKind);
    case TypeCategory::Real:
    case TypeCategory::Complex:
      return std::make_pair(yCat, yKind);
    default:
      break;
    }
    break;
  case TypeCategory::Real:
    switch (yCat) {
    case TypeCategory::Integer:
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
      return std::nullopt;
    }
  case TypeCategory::Logical:
    if (yCat == TypeCategory::Logical) {
      return std::make_pair(TypeCategory::Logical, maxKind);
    } else {
      return std::nullopt;
    }
  default:
    break;
  }
  return std::nullopt;
}

} // namespace Fortran::runtime
#endif // FORTRAN_RUNTIME_TOOLS_H_
