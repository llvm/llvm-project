//===----------------------- cxa_thread_atexit.cpp ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "cxxabi.h"

namespace __cxxabiv1 {
extern "C" {

#ifdef HAVE___CXA_THREAD_ATEXIT_IMPL

_LIBCXXABI_FUNC_VIS int __cxa_thread_atexit(void (*dtor)(void *), void *obj,
                                            void *dso_symbol) throw() {
  extern int __cxa_thread_atexit_impl(void (*)(void *), void *, void *);
  return __cxa_thread_atexit_impl(dtor, obj, dso_symbol);
}

#endif // HAVE__CXA_THREAD_ATEXIT_IMPL

} // extern "C"
} // namespace __cxxabiv1
