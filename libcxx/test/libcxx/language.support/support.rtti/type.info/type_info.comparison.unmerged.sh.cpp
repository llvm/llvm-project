//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-rtti

// The executable references the shared library by its host path, which doesn't exist on the device.
// UNSUPPORTED: android

// The type A has external linkage, so its RTTI name is not prefixed with '*'.
// Its RTTI is emitted separately in the shared library and in the main executable,
// and libc++ compares the names with strcmp, so the two type_index values must be equal.

// RUN: %{cxx} %s %{flags} %{compile_flags} -fPIC -shared -o %t.lib.so -DTU1 -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2 %{link_flags}
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.main.o -DMAIN -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2
// RUN: %{cxx} %t.main.o %t.lib.so %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include <cassert>
#include <typeindex>

// Hidden visibility keeps each copy of the RTTI local to its own module.
struct [[gnu::visibility("hidden")]] A {
  int x;
};

std::type_index registerA();

#if defined(TU1)
std::type_index registerA() { return std::type_index(typeid(A)); }

#elif defined(MAIN)
int main(int, char**) {
  std::type_index executable = std::type_index(typeid(A));
  std::type_index shared     = registerA();

  assert(executable == shared);
  return 0;
}
#else
#  error
#endif
