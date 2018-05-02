//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
//
// <atomic>

// Test that including <atomic> fails to compile when we want to use C atomics
// in C++ and have corresponding macro defined.

// MODULES_DEFINES: __ALLOW_STDC_ATOMICS_IN_CXX__
#ifndef __ALLOW_STDC_ATOMICS_IN_CXX__
#define __ALLOW_STDC_ATOMICS_IN_CXX__
#endif

#include <atomic>
// expected-error@atomic:* {{<stdatomic.h> is incompatible with the C++ standard library}}

int main()
{
}

