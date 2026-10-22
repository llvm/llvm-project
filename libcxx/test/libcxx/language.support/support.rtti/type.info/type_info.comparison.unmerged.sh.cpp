//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-rtti

// The Microsoft ABI type_info can't be constructed from a name.
// UNSUPPORTED: msvc

// Test the non-unique (strcmp) type_info comparison with unmerged RTTI,
// two type_info objects that for the same type whose names are equal strings
// at different addresses (which occurs with shared library boundaries).
// Unmerged RTTI is simulated directly here, without any shared library needed.

// RUN: %{cxx} %s %{flags} %{compile_flags} -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2 %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include <cassert>
#include <cstring>
#include <typeinfo>

// std::type_info's name constructor is protected, so we derive from it to expose a public constructor.
struct TypeInfo : std::type_info {
  explicit TypeInfo(const char* name) : std::type_info(name) {}
};

static const char external_1[] = "1A";
static const char external_2[] = "1A";
static const char internal_1[] = "*N12_GLOBAL__N_11AE";
static const char internal_2[] = "*N12_GLOBAL__N_11AE";

int main(int, char**) {
  // Externally visible type: one type, so copies compare equal by name
  TypeInfo ext1(external_1);
  TypeInfo ext2(external_2);
  assert(ext1 == ext2);
  assert(!ext1.before(ext2) && !ext2.before(ext1));

  // Internal-linkage type ('*' prefix): equal names does not mean equal type.
  // so only the RTTI object compares equal.
  TypeInfo int1(internal_1);
  TypeInfo int2(internal_2);
  assert(int1 == int1);
  assert(int1 != int2);
  assert(int1.before(int2) != int2.before(int1));

  assert(std::strcmp(int1.name(), "N12_GLOBAL__N_11AE") == 0);
  return 0;
}
