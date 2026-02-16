//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test relies on `typeid` and thus requires `-frtti`.
// UNSUPPORTED: no-rtti

// Make sure that we don't get ODR violations with __exception_guard when
// linking together TUs compiled with different values of -f[no-]exceptions.

// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.except.o   -O1 -fexceptions
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.noexcept.o -O1 -fno-exceptions
// RUN: %{cxx} %{flags} %{link_flags} -o %t.exe %t.except.o %t.noexcept.o
// RUN: %{run}

#include <__cxx03/__utility/exception_guard.h>
#include <cassert>
#include <cstring>
#include <typeinfo>

struct Rollback {
  void operator()() {}
};

#if defined(__cpp_exceptions) && __cpp_exceptions >= 199711L

const char* func();

int main(int, char**) {
  assert(std::strcmp(typeid(std::__exception_guard<Rollback>).name(), func()) != 0);

  return 0;
}

#else

const char* func() { return typeid(std::__exception_guard<Rollback>).name(); }

#endif
