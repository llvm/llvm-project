//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that the non-unique (strcmp-based) type_info comparison considers two copies of the
// RTTI for the same type to be equal. Two copies are created by giving a type hidden visibility
// and using it from both a shared library and the executable that loads it.

// UNSUPPORTED: no-rtti

// Windows needs explicit symbol exports
// UNSUPPORTED: windows

// Android executable looks for the library at its host path, which doesn't exist on the device
// UNSUPPORTED: android

// picolibc and LLVM libc only produce static executables
// UNSUPPORTED: LIBCXX-PICOLIBC-FIXME, LLVM-LIBC-FIXME

// z/OS links against DLLs through a side deck.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

// RUN: %{cxx} %s %{flags} %{compile_flags} -fPIC -shared -o %t.lib.so -DTU1 -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2 %if darwin || target={{.+}}-aix{{.*}} %{ %{link_flags} %} %else %{ -nostdlib++ %}
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

  assert(executable.name() != shared.name());
  assert(executable == shared);
  assert(!(executable < shared));
  assert(!(shared < executable));
  return 0;
}
#else
#  error
#endif
