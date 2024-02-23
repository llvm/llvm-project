//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new(std::size_t, const std::nothrow_t&);
// void* operator new(std::size_t, std::align_val_t, const std::nothrow_t&);
// void* operator new[](std::size_t, const std::nothrow_t&);
// void* operator new[](std::size_t, std::align_val_t, const std::nothrow_t&);

// This test ensures that we catch the case where `new` has been overridden but `new(nothrow)`
// has not been overridden, and the library is compiled with -fno-exceptions.
//
// In that case, it is impossible for libc++ to provide a Standards conforming implementation
// of `new(nothrow)`, so the only viable option is to terminate the program.

// REQUIRES: has-unix-headers
// UNSUPPORTED: c++03

// We only know how to diagnose this on platforms that use the ELF or Mach-O object file formats.
// XFAIL: target={{.+}}-windows-{{.+}}

// TODO: We currently don't have a way to express that the built library was
//       compiled with -fno-exceptions, so if the library was built with support
//       for exceptions but we run the test suite without exceptions, this will
//       spuriously fail.
// REQUIRES: no-exceptions

#include <cstddef>
#include <new>

#include "check_assertion.h"

// Override the throwing versions of operator new, but not the nothrow versions.
alignas(32) char DummyData[32 * 3];
void* operator new(std::size_t) { return DummyData; }
void* operator new(std::size_t, std::align_val_t) { return DummyData; }
void* operator new[](std::size_t) { return DummyData; }
void* operator new[](std::size_t, std::align_val_t) { return DummyData; }

void operator delete(void*) noexcept {}
void operator delete(void*, std::align_val_t) noexcept {}
void operator delete[](void*) noexcept {}
void operator delete[](void*, std::align_val_t) noexcept {}

int main(int, char**) {
  std::size_t size       = 3;
  std::align_val_t align = static_cast<std::align_val_t>(32);
  EXPECT_ANY_DEATH((void)operator new(size, std::nothrow));
  EXPECT_ANY_DEATH((void)operator new(size, align, std::nothrow));
  EXPECT_ANY_DEATH((void)operator new[](size, std::nothrow));
  EXPECT_ANY_DEATH((void)operator new[](size, align, std::nothrow));

  return 0;
}
