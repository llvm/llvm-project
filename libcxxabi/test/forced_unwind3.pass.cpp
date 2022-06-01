//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Let's run ForcedUnwind until it reaches end of the stack, this test simulates
// what pthread_cancel does.

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads
// UNSUPPORTED: no-exceptions

#include <assert.h>
#include <exception>
#include <stdlib.h>
#include <string.h>
#include <unwind.h>
#include <thread>
#include <tuple>
#include <__cxxabi_config.h>

// TODO: dump version back to 14 once clang is updated on the CI.
#if defined(_LIBCXXABI_ARM_EHABI) && defined(__clang__) && __clang_major__ < 15
// _Unwind_ForcedUnwind is not available or broken before version 14.
int main(int, char**) { return 0; }

#else
static bool destructorCalled = false;

struct myClass {
  myClass() {}
  ~myClass() {
    assert(destructorCalled == false);
    destructorCalled = true;
  };
};

template <typename T>
struct Stop;

template <typename R, typename... Args>
struct Stop<R (*)(Args...)> {
  // The third argument of _Unwind_Stop_Fn is uint64_t in Itanium C++ ABI/LLVM
  // libunwind while _Unwind_Exception_Class in libgcc.
  typedef typename std::tuple_element<2, std::tuple<Args...>>::type type;

  static _Unwind_Reason_Code stop(int, _Unwind_Action actions, type, struct _Unwind_Exception*, struct _Unwind_Context*,
                                  void*) {
    if (actions & _UA_END_OF_STACK) {
      assert(destructorCalled == true);
      exit(0);
    }
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

__attribute__((__noinline__)) static void test() {
  myClass c{};
  forced_unwind();
  abort();
}

int main(int, char**) {
  std::thread t{test};
  t.join();
  return -1;
}
#endif
