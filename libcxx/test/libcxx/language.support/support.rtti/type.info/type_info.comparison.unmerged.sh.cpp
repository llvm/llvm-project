//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the strcmp-based type_info comparison treats two copies of the RTTI
// for the same type as equal. A has hidden visibility, so the shared library and
// the executable each get their own copy.

// UNSUPPORTED: no-rtti
// UNSUPPORTED: android

// XFAIL: target={{.+}}-zos{{.*}}

// REQUIRES: can-build-shared-libraries

// RUN: %{cxx} %s %{flags} %{compile_flags} -fPIC -c -o %t.lib.o -DLIB -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2
// RUN: %{cxx} %t.lib.o %{flags} %{link_flags} -shared -o %t.lib.so
// RUN: %{build} -DMAIN -D_LIBCPP_TYPEINFO_COMPARISON_IMPLEMENTATION=2 %t.lib.so
// RUN: %{exec} %t.exe

#include <cassert>
#include <typeindex>

struct __attribute__((visibility("hidden"))) A {};

std::type_index from_library();

#if defined(LIB)
std::type_index from_library() { return typeid(A); }
#elif defined(MAIN)
int main(int, char**) {
  std::type_index from_executable = typeid(A);
  assert(from_executable.name() != from_library().name());
  assert(from_executable == from_library());
  assert(!(from_executable < from_library()));
  assert(!(from_library() < from_executable));
  return 0;
}
#else
#  error
#endif
