//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test libc++'s implementation of align_val_t, and the relevant new/delete
// overloads in all dialects when -faligned-allocation is present.

// Libc++ defers to the underlying MSVC library to provide the new/delete
// definitions, which does not yet provide aligned allocation
// XFAIL: LIBCXX-WINDOWS-FIXME

// The dylibs shipped before macosx10.13 do not contain the aligned allocation
// functions, so trying to force using those with -faligned-allocation results
// in a link error.
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}}

// REQUIRES: -faligned-allocation
// ADDITIONAL_COMPILE_FLAGS: -faligned-allocation

#include <new>
#include <typeinfo>
#include <string>
#include <cassert>

#include "test_macros.h"

int main(int, char**) {
  {
    static_assert(std::is_enum<std::align_val_t>::value, "");
    typedef std::underlying_type<std::align_val_t>::type UT;
    static_assert((std::is_same<UT, std::size_t>::value), "");
  }
  {
    static_assert((!std::is_constructible<std::align_val_t, std::size_t>::value), "");
#if TEST_STD_VER >= 11
    static_assert(!std::is_constructible<std::size_t, std::align_val_t>::value, "");
#else
    static_assert((std::is_constructible<std::size_t, std::align_val_t>::value), "");
#endif
  }
  {
    std::align_val_t a = std::align_val_t(0);
    std::align_val_t b = std::align_val_t(32);
    assert(a != b);
    assert(a == std::align_val_t(0));
    assert(b == std::align_val_t(32));
  }
  {
    void *ptr = ::operator new(1, std::align_val_t(128));
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 128 == 0);
    ::operator delete(ptr, std::align_val_t(128));
  }
  {
    void *ptr = ::operator new(1, std::align_val_t(128), std::nothrow);
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 128 == 0);
    ::operator delete(ptr, std::align_val_t(128), std::nothrow);
  }
  {
    void *ptr = ::operator new[](1, std::align_val_t(128));
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 128 == 0);
    ::operator delete[](ptr, std::align_val_t(128));
  }
  {
    void *ptr = ::operator new[](1, std::align_val_t(128), std::nothrow);
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % 128 == 0);
    ::operator delete[](ptr, std::align_val_t(128), std::nothrow);
  }
#ifndef TEST_HAS_NO_RTTI
  {
    // Check that libc++ doesn't define align_val_t in a versioning namespace.
    // And that it mangles the same in C++03 through C++17
    assert(typeid(std::align_val_t).name() == std::string("St11align_val_t"));
  }
#endif

  return 0;
}
