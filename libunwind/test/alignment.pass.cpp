// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// The Itanium ABI requires that _Unwind_Exception objects are "double-word
// aligned".

#include <unwind.h>

struct MaxAligned {} __attribute__((aligned));
static_assert(alignof(_Unwind_Exception) == alignof(MaxAligned), "");

int main()
{
}
