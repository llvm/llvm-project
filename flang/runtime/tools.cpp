//===-- runtime/tools.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "tools.h"
#include "terminator.h"
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
    std::memcpy(p, s, length);
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
    std::memcpy(to, from, len);
    std::memset(to + len, ' ', toLength - len);
  } else {
    std::memcpy(to, from, toLength);
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
    terminator.Crash(
        "not yet implemented: %s: KIND=%d argument", intrinsic, kind);
  }
}

RT_API_ATTRS void ShallowCopyDiscontiguousToDiscontiguous(
    const Descriptor &to, const Descriptor &from) {
  SubscriptValue toAt[maxRank], fromAt[maxRank];
  to.GetLowerBounds(toAt);
  from.GetLowerBounds(fromAt);
  std::size_t elementBytes{to.ElementBytes()};
  for (std::size_t n{to.Elements()}; n-- > 0;
       to.IncrementSubscripts(toAt), from.IncrementSubscripts(fromAt)) {
    std::memcpy(
        to.Element<char>(toAt), from.Element<char>(fromAt), elementBytes);
  }
}

RT_API_ATTRS void ShallowCopyDiscontiguousToContiguous(
    const Descriptor &to, const Descriptor &from) {
  char *toAt{to.OffsetElement()};
  SubscriptValue fromAt[maxRank];
  from.GetLowerBounds(fromAt);
  std::size_t elementBytes{to.ElementBytes()};
  for (std::size_t n{to.Elements()}; n-- > 0;
       toAt += elementBytes, from.IncrementSubscripts(fromAt)) {
    std::memcpy(toAt, from.Element<char>(fromAt), elementBytes);
  }
}

RT_API_ATTRS void ShallowCopyContiguousToDiscontiguous(
    const Descriptor &to, const Descriptor &from) {
  SubscriptValue toAt[maxRank];
  to.GetLowerBounds(toAt);
  char *fromAt{from.OffsetElement()};
  std::size_t elementBytes{to.ElementBytes()};
  for (std::size_t n{to.Elements()}; n-- > 0;
       to.IncrementSubscripts(toAt), fromAt += elementBytes) {
    std::memcpy(to.Element<char>(toAt), fromAt, elementBytes);
  }
}

RT_API_ATTRS void ShallowCopy(const Descriptor &to, const Descriptor &from,
    bool toIsContiguous, bool fromIsContiguous) {
  if (toIsContiguous) {
    if (fromIsContiguous) {
      std::memcpy(to.OffsetElement(), from.OffsetElement(),
          to.Elements() * to.ElementBytes());
    } else {
      ShallowCopyDiscontiguousToContiguous(to, from);
    }
  } else {
    if (fromIsContiguous) {
      ShallowCopyContiguousToDiscontiguous(to, from);
    } else {
      ShallowCopyDiscontiguousToDiscontiguous(to, from);
    }
  }
}

RT_API_ATTRS void ShallowCopy(const Descriptor &to, const Descriptor &from) {
  ShallowCopy(to, from, to.IsContiguous(), from.IsContiguous());
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime
