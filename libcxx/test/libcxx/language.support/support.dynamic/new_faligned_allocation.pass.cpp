//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test libc++'s implementation of align_val_t, and the relevant new/delete
// overloads in all dialects when -faligned-allocation is present.

// Libc++ when built for z/OS doesn't contain the aligned allocation functions,
// nor does the dynamic library shipped with z/OS.
// UNSUPPORTED: target={{.+}}-zos{{.*}}

// REQUIRES: -faligned-allocation
// ADDITIONAL_COMPILE_FLAGS: -faligned-allocation

#include <cassert>
#include <new>
#include <string>
#include <type_traits>
#include <typeinfo>

#include "test_macros.h"

static void test_allocations(std::size_t size, size_t alignment) {
  {
    void* ptr = ::operator new(size, std::align_val_t(alignment));
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0);
    ::operator delete(ptr, std::align_val_t(alignment));
  }
  {
    void* ptr = ::operator new(size, std::align_val_t(alignment), std::nothrow);
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0);
    ::operator delete(ptr, std::align_val_t(alignment), std::nothrow);
  }
  {
    void* ptr = ::operator new[](size, std::align_val_t(alignment));
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0);
    ::operator delete[](ptr, std::align_val_t(alignment));
  }
  {
    void* ptr = ::operator new[](size, std::align_val_t(alignment), std::nothrow);
    assert(ptr);
    assert(reinterpret_cast<std::uintptr_t>(ptr) % alignment == 0);
    ::operator delete[](ptr, std::align_val_t(alignment), std::nothrow);
  }
}

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
  // First, check the basic case, a large allocation with alignment==size.
  test_allocations(64, 64);
  // Size being a multiple of alignment also needs to be supported.
  test_allocations(64, 32);
  // When aligned allocation is implemented using posix_memalign,
  // that function requires a minimum alignment of sizeof(void*).
  // Check that we can also create overaligned allocations with
  // an alignment argument less than sizeof(void*).
  test_allocations(2, 2);
  // When implemented using the C11 aligned_alloc() function,
  // that requires that size be a multiple of alignment.
  // However, the C++ operator new has no such requirements.
  // Check that we can create an overaligned allocation that does
  // adhere to not have this constraint.
  test_allocations(1, 128);
  // Finally, test size > alignment, but with size not being
  // a multiple of alignment.
  test_allocations(65, 32);
#ifndef TEST_HAS_NO_RTTI
  {
    // Check that libc++ doesn't define align_val_t in a versioning namespace.
    // And that it mangles the same in C++03 through C++17
#ifdef _MSC_VER
    // MSVC uses a different C++ ABI with a different name mangling scheme.
    // The type id name doesn't seem to contain the mangled form at all.
    assert(typeid(std::align_val_t).name() == std::string("enum std::align_val_t"));
#else
    assert(typeid(std::align_val_t).name() == std::string("St11align_val_t"));
#endif
  }
#endif

  return 0;
}
