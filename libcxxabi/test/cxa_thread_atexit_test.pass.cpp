//===--------------------- cxa_thread_atexit_test.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcxxabi-no-threads
// REQUIRES: linux

// this test will only work if CMake detects a real __cxa_thread_atexit_impl
// at configure time. This function, however, was added only in glibc 2.18,
// and there are still plenty of systems only using 2.17 (Ex RHEL 7).
// XFAIL: libcxxabi-no-cxa-thread-atexit-impl

#include <assert.h>
#include <cxxabi.h>

static bool AtexitImplCalled = false;

extern "C" int __cxa_thread_atexit_impl(void (*dtor)(void *), void *obj,
                                        void *dso_symbol) {
  assert(dtor == reinterpret_cast<void (*)(void *)>(1));
  assert(obj == reinterpret_cast<void *>(2));
  assert(dso_symbol == reinterpret_cast<void *>(3));
  AtexitImplCalled = true;
  return 4;
}

int main() {
  int RV = __cxxabiv1::__cxa_thread_atexit(
      reinterpret_cast<void (*)(void *)>(1), reinterpret_cast<void *>(2),
      reinterpret_cast<void *>(3));
  assert(RV == 4);
  assert(AtexitImplCalled);
  return 0;
}
