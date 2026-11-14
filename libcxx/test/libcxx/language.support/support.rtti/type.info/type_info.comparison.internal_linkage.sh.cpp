//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-rtti

// Older versions don't mark RTTI names of types with internal linkage with a '*'
// XFAIL: clang-{{2[1-3]}}, apple-clang

// Itanium ABI implementations prefix the RTTI name of a type with internal linkage with a '*',
// which lets this implementation distinguish between two internally linked types.

// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.tu1.o -DTU1 -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.tu2.o -DTU2 -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.main.o -DMAIN -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2
// RUN: %{cxx} %t.tu1.o %t.tu2.o %t.main.o %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include <cassert>
#include <typeindex>
#include <vector>

extern std::vector<std::type_index> registry;

void register1();
void register2();

#if defined(TU1)
namespace {
struct A {
  bool x;
};
} // namespace
void register1() { registry.push_back(std::type_index(typeid(A))); }
#elif defined(TU2)
namespace {
struct A {
  int x, y;
};
} // namespace
void register2() { registry.push_back(std::type_index(typeid(A))); }
#elif defined(MAIN)
std::vector<std::type_index> registry;

int main(int, char**) {
  register1();
  register2();

  assert(registry.size() == 2);
  assert(registry[0] != registry[1]);
  return 0;
}
#else
#  error
#endif
