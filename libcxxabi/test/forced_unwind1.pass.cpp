//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// _Unwind_ForcedUnwind raised exception can be caught by catch (...) and be
// rethrown. If not rethrown, exception_cleanup will be called.

// UNSUPPORTED: no-exceptions, c++03

// These tests fail on previously released dylibs, investigation needed.
// XFAIL: with_system_cxx_lib=macosx10.15
// XFAIL: with_system_cxx_lib=macosx10.14
// XFAIL: with_system_cxx_lib=macosx10.13
// XFAIL: with_system_cxx_lib=macosx10.12
// XFAIL: with_system_cxx_lib=macosx10.11
// XFAIL: with_system_cxx_lib=macosx10.10
// XFAIL: with_system_cxx_lib=macosx10.9

#include <stdlib.h>
#include <string.h>
#include <unwind.h>
#include <tuple>
#include <__cxxabi_config.h>

#if defined(_LIBCXXABI_ARM_EHABI)
int main(int, char**) {
  return 0;
}
#else
static int bits = 0;

struct C {
  int bit;
  C(int b) : bit(b) {}
  ~C() { bits |= bit; }
};

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

static void cleanup(_Unwind_Reason_Code, struct _Unwind_Exception* exc) {
  bits |= 8;
  delete exc;
}

static void forced_unwind() {
  _Unwind_Exception* exc = new _Unwind_Exception;
  exc->exception_class = 0;
  exc->exception_cleanup = cleanup;
  _Unwind_ForcedUnwind(exc, Stop<_Unwind_Stop_Fn>::stop, 0);
  abort();
}

static void test() {
  try {
    C four(4);
    try {
      C one(1);
      forced_unwind();
    } catch (...) {
      bits |= 2;
      throw;
    }
  } catch (int) {
  } catch (...) {
    // __cxa_end_catch calls cleanup.
  }
}

int main(int, char**) {
  test();
  return bits != 15;
}
#endif
