//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// aligned allocation functions are not provided prior to macosx10.13
// UNSUPPORTED: macosx10.12
// UNSUPPORTED: macosx10.11
// UNSUPPORTED: macosx10.10
// UNSUPPORTED: macosx10.9
// UNSUPPORTED: macosx10.8
// UNSUPPORTED: macosx10.7

#include <new>


#ifdef _LIBCPP_HAS_NO_ALIGNED_ALLOCATION
#   error "libc++ should have aligned allocation in C++17 and up when targeting a platform that supports it"
#endif

int main() { }
