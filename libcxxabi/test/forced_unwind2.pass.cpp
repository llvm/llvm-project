//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Forced unwinding causes std::terminate when going through noexcept.

// UNSUPPORTED: no-exceptions, c++03

// VE only supports SjLj and doesn't provide _Unwind_ForcedUnwind.
// UNSUPPORTED: target={{ve-.*}}

// These tests fail on previously released dylibs, investigation needed.
// XFAIL: stdlib=system && target={{.+}}-apple-macosx10.{{9|10|11|12|13|14|15}}
// XFAIL: stdlib=system && target={{.+}}-apple-macosx{{11.0|12.0}}

#include <exception>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unwind.h>
#include <tuple>
#include <__cxxabi_config.h>

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
  memset(&exc->exception_class, 0, sizeof(exc->exception_class));
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
