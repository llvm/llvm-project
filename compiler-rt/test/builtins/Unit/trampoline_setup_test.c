// RUN: %clang_builtins %s %librt -fnested-functions -o %t && %run %t
/* ===-- trampoline_setup_test.c - Test __trampoline_setup -----------------===
 *
 *                     The LLVM Compiler Infrastructure
 *
 * This file is dual licensed under the MIT and the University of Illinois Open
 * Source Licenses. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 */


#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <sys/mman.h>

/*
 * Tests nested functions
 * The ppc compiler generates a call to __trampoline_setup
 * The i386 and x86_64 compilers generate a call to ___enable_execute_stack
 */

/*
 * Note that, nested functions are not ISO C and are not supported in Clang.
 */

#if !defined(__clang__)

typedef int (*nested_func_t)(int x);

nested_func_t proc;

int main() {
    /* Some locals */
    int c = 10;
    int d = 7;
    
    /* Define a nested function: */
    int bar(int x) { return x*5 + c*d; };

    /* Assign global to point to nested function
     * (really points to trampoline). */
    proc = bar;
    
    /* Invoke nested function: */
    c = 4;
    if ( (*proc)(3) != 43 )
        return 1;
    d = 5;
    if ( (*proc)(4) != 40 )
        return 1;

    /* Success. */
    return 0;
}

#else

int main() {
    printf("skipped\n");
    return 0;
}

#endif
