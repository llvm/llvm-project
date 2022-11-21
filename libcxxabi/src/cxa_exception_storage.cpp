//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//
//  This file implements the storage for the "Caught Exception Stack"
//  https://itanium-cxx-abi.github.io/cxx-abi/abi-eh.html#cxx-exc-stack
//
//===----------------------------------------------------------------------===//

#include "cxa_exception.h"

namespace __cxxabiv1 {

#if defined(_LIBCXXABI_HAS_NO_THREADS)

extern "C" {
    static __cxa_eh_globals eh_globals;
    __cxa_eh_globals *__cxa_get_globals() { return &eh_globals; }
    __cxa_eh_globals *__cxa_get_globals_fast() { return &eh_globals; }
} // extern "C"

#else

namespace {
    __cxa_eh_globals *__globals() {
        static thread_local __cxa_eh_globals eh_globals;
        return &eh_globals;
    }
} // namespace

extern "C" {
    __cxa_eh_globals *__cxa_get_globals() { return __globals(); }
    __cxa_eh_globals *__cxa_get_globals_fast() { return __globals(); }
} // extern "C"

#endif

} // namespace __cxxabiv1
