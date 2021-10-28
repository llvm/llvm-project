//===-------------------- test_exception_storage.cpp ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO(ldionne): This test fails on Ubuntu Focal on our CI nodes (and only there), in 32 bit mode.
// UNSUPPORTED: linux && 32bits-on-64bits

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <__threading_support>
#include <unistd.h>

#include "../src/cxa_exception.h"

typedef __cxxabiv1::__cxa_eh_globals globals_t ;

void *thread_code (void *parm) {
    size_t *result = (size_t *) parm;
    globals_t *glob1, *glob2;

    glob1 = __cxxabiv1::__cxa_get_globals ();
    if ( NULL == glob1 )
        std::printf("Got null result from __cxa_get_globals\n");

    glob2 = __cxxabiv1::__cxa_get_globals_fast ();
    if ( glob1 != glob2 )
        std::printf("Got different globals!\n");

    *result = (size_t) glob1;
#ifndef _LIBCXXABI_HAS_NO_THREADS
    sleep ( 1 );
#endif
    return parm;
}

#ifndef _LIBCXXABI_HAS_NO_THREADS
#define NUMTHREADS  10
size_t                 thread_globals [ NUMTHREADS ] = { 0 };
std::__libcpp_thread_t   threads        [ NUMTHREADS ];
#endif

int main () {
    int retVal = 0;

#ifndef _LIBCXXABI_HAS_NO_THREADS
//  Make the threads, let them run, and wait for them to finish
    for ( int i = 0; i < NUMTHREADS; ++i )
        std::__libcpp_thread_create ( threads + i, thread_code, (void *) (thread_globals + i));
    for ( int i = 0; i < NUMTHREADS; ++i )
        std::__libcpp_thread_join ( &threads [ i ] );

    for ( int i = 0; i < NUMTHREADS; ++i ) {
        if ( 0 == thread_globals [ i ] ) {
            std::printf("Thread #%d had a zero global\n", i);
            retVal = 1;
        }
    }

    std::sort ( thread_globals, thread_globals + NUMTHREADS );
    for ( int i = 1; i < NUMTHREADS; ++i ) {
        if ( thread_globals [ i - 1 ] == thread_globals [ i ] ) {
            std::printf("Duplicate thread globals (%d and %d)\n", i-1, i);
            retVal = 2;
        }
    }
#else // _LIBCXXABI_HAS_NO_THREADS
    size_t thread_globals;
    // Check that __cxa_get_globals() is not NULL.
    if (thread_code(&thread_globals) == 0) {
        retVal = 1;
    }
#endif // !_LIBCXXABI_HAS_NO_THREADS
    return retVal;
}
