//===--- TypeTraits.cpp - Type Traits Support -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the type traits support functions.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/TypeTraits.h"
#include <cassert>
#include <cstring>
using namespace clang;

// static constexpr const char *TypeTraitNames[] = {...};
// static constexpr const char *TypeTraitSpellings[] = {...};
// static constexpr const char *ArrayTypeTraitNames[] = {...};
// static constexpr const char *ArrayTypeTraitSpellings[] = {...};
// static constexpr const char *UnaryExprOrTypeTraitNames[] = {...};
// static constexpr const char *UnaryExprOrTypeTraitSpellings[] = {...};
// static constexpr const unsigned TypeTraitArities[] = {...};
#define EMIT_ARRAYS
#include "clang/Basic/Traits.inc"

const char *clang::getTraitName(TypeTrait T) {
  assert(T <= TT_Last && "invalid enum value!");
  return TypeTraitNames[T];
}

const char *clang::getTraitName(ArrayTypeTrait T) {
  assert(T <= ATT_Last && "invalid enum value!");
  return ArrayTypeTraitNames[T];
}

const char *clang::getTraitName(UnaryExprOrTypeTrait T) {
  assert(T <= UETT_Last && "invalid enum value!");
  return UnaryExprOrTypeTraitNames[T];
}

const char *clang::getTraitSpelling(TypeTrait T) {
  assert(T <= TT_Last && "invalid enum value!");
  if (T == BTT_IsDeducible) {
    // The __is_deducible is an internal-only type trait. To hide it from
    // external users, we define it with an empty spelling name, preventing the
    // clang parser from recognizing its token kind.
    // However, other components such as the AST dump still require the real
    // type trait name. Therefore, we return the real name when needed.
    assert(std::strlen(TypeTraitSpellings[T]) == 0);
    return "__is_deducible";
  }
  return TypeTraitSpellings[T];
}

const char *clang::getTraitSpelling(ArrayTypeTrait T) {
  assert(T <= ATT_Last && "invalid enum value!");
  return ArrayTypeTraitSpellings[T];
}

const char *clang::getTraitSpelling(UnaryExprOrTypeTrait T) {
  assert(T <= UETT_Last && "invalid enum value!");
  return UnaryExprOrTypeTraitSpellings[T];
}

unsigned clang::getTypeTraitArity(TypeTrait T) {
  assert(T <= TT_Last && "invalid enum value!");
  return TypeTraitArities[T];
}
