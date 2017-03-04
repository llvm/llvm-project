//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcxxabi-no-exceptions
// UNSUPPORTED: c++98, c++03

// The system unwind.h on OS X provides an incorrectly aligned _Unwind_Exception
// type. That causes these tests to fail. This XFAIL is my best attempt at
// working around this failure.
// XFAIL: darwin && libcxxabi-has-system-unwinder

// Test that the address of the exception object is properly aligned to the
// largest supported alignment for the system.

#include <cstdint>
#include <cassert>

#include <unwind.h>

struct __attribute__((aligned)) AlignedType {};
static_assert(alignof(AlignedType) == alignof(_Unwind_Exception),
  "_Unwind_Exception is incorrectly aligned. This test is expected to fail");

struct MinAligned {  };
static_assert(alignof(MinAligned) == 1 && sizeof(MinAligned) == 1, "");

int main() {
  for (int i=0; i < 10; ++i) {
    try {
      throw MinAligned{};
    } catch (MinAligned const& ref) {
      assert(reinterpret_cast<uintptr_t>(&ref) % alignof(AlignedType) == 0);
    }
  }
}
