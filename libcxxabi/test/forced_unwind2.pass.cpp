//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Forced unwinding causes std::terminate when going through noexcept.

// UNSUPPORTED: no-exceptions, c++03

// These tests fail on previously released dylibs, investigation needed.
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.15
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.14
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.13
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.12
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9

#include <exception>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unwind.h>
#include <tuple>
#include <__cxxabi_config.h>

#if defined(_LIBCXXABI_ARM_EHABI)
int main(int, char**) {
  return 0;
}
#else
template <typename T>
struct Stop;

template <typename R, typename... Args>
struct Stop<R (*)(Args...)> {
  // The third argument of _Unwind_Stop_Fn is uint64_t in Itanium C++ ABI/LLVM
  // libunwind while _Unwind_Exception_Class in libgcc.
  typedef typename std::tuple_element<2, std::tuple<Args...>>::type type;

  static _Unwind_Reason_Code stop(int, _Unwind_Action actions, type,
                                  struct _Unwind_Exception*,
                                  struct _Unwind_Context*, void*) {
    if (actions & _UA_END_OF_STACK)
      abort();
    return _URC_NO_REASON;
  }
};

static void forced_unwind() {
  _Unwind_Exception* exc = new _Unwind_Exception;
  exc->exception_class = 0;
  exc->exception_cleanup = 0;
  _Unwind_ForcedUnwind(exc, Stop<_Unwind_Stop_Fn>::stop, 0);
  abort();
}

static void test() noexcept { forced_unwind(); }

static void terminate() { exit(0); }

int main(int, char**) {
  std::set_terminate(terminate);
  try {
    test();
  } catch (...) {
  }
  abort();
}
#endif
