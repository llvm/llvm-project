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

// <mutex>

// template <class ...Mutex> class lock_guard;

// Test that the variadic lock guard implementation compiles in all standard
// dialects, including C++03, even though it is forward declared using
// variadic templates.

#define _LIBCPP_ABI_VARIADIC_LOCK_GUARD
#include "mutex.pass.cpp" // Use the existing non-variadic test
