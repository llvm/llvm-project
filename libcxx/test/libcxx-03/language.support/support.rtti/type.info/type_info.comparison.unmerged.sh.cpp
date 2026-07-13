//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-rtti

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
  namespace { struct A { bool x; }; }
  void register1() { registry.push_back(std::type_index(typeid(A))); }
#elif defined(TU2)
  namespace { struct A { int x, y; }; }
  void register2() { registry.push_back(std::type_index(typeid(A))); }
#elif defined(MAIN)
  std::vector<std::type_index> registry;

  int main(int, char**) {
    register1();
    register2();

    assert(registry.size() == 2);
    assert(registry[0] == registry[1]);
    return 0;
  }
#else
# error
#endif
