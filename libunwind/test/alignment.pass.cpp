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

// EHABI  : 8-byte aligned
// itanium: largest supported alignment for the system
#if defined(_LIBUNWIND_ARM_EHABI)
static_assert(alignof(_Unwind_Control_Block) == 8,
              "_Unwind_Control_Block must be double-word aligned");
#else
struct MaxAligned {} __attribute__((__aligned__));
static_assert(alignof(_Unwind_Exception) == alignof(MaxAligned),
              "_Unwind_Exception must be maximally aligned");
#endif

int main()
{
}
