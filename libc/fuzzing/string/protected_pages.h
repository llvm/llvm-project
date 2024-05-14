//===-- protected_pages.h -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file provides protected pages that fault when accessing prior or past
// it. This is useful to check memory functions that must not access outside of
// the provided size limited buffer.
//===----------------------------------------------------------------------===//

#ifndef LIBC_FUZZING_STRING_PROTECTED_PAGES_H
#define LIBC_FUZZING_STRING_PROTECTED_PAGES_H

#include <stddef.h>   // size_t
#include <stdint.h>   // uint8_t
#include <sys/mman.h> // mmap, munmap
#include <unistd.h>   // sysconf, _SC_PAGESIZE

// Returns mmap page size.
size_t GetPageSize() { return sysconf(_SC_PAGESIZE); }

// Represents a page of memory whose access can be configured throught the
// 'WithAccess' function. Accessing data above or below this page will trap as
// it is sandwiched between two pages with no read / write access.
struct Page {
  // Returns an aligned pointer that can be accessed up to page_size. Accessing
  // data at ptr[-1] will fault.
  uint8_t *bottom(size_t size) const {
    if (size >= page_size)
      __builtin_trap();
    return page_ptr;
  }
  // Returns a pointer to a buffer that can be accessed up to size. Accessing
  // data at ptr[size] will trap.
  uint8_t *top(size_t size) const { return page_ptr + page_size - size; }

  // protection is one of PROT_READ / PROT_WRITE.
  Page &WithAccess(int protection) {
    if (mprotect(page_ptr, page_size, protection) != 0)
      __builtin_trap();
    return *this;
  }

  const size_t page_size;
  uint8_t *const page_ptr;
};

// Allocates 5 consecutive pages that will trap if accessed.
// | page layout | access | page name |
// |-------------|--------|:---------:|
// | 1           | trap   |           |
// | 2           | custom |     A     |
// | 3           | trap   |           |
// | 4           | custom |     B     |
// | 5           | trap   |           |
//
// The pages A and B can be retrieved as with 'GetPageA' / 'GetPageB' and their
// accesses can be customized through the 'WithAccess' function.
struct ProtectedPages {
  static constexpr size_t PAGES = 5;

  ProtectedPages()
      : page_size(GetPageSize()),
        ptr(mmap(/*address*/ nullptr, /*length*/ PAGES * page_size,
                 /*protection*/ PROT_NONE,
                 /*flags*/ MAP_PRIVATE | MAP_ANONYMOUS, /*fd*/ -1,
                 /*offset*/ 0)) {
    if (reinterpret_cast<intptr_t>(ptr) == -1)
      __builtin_trap();
  }
  ~ProtectedPages() { munmap(ptr, PAGES * page_size); }

  Page GetPageA() const { return Page{page_size, page<1>()}; }
  Page GetPageB() const { return Page{page_size, page<3>()}; }

private:
  template <size_t index> uint8_t *page() const {
    static_assert(index < PAGES);
    return static_cast<uint8_t *>(ptr) + (index * page_size);
  }

  const size_t page_size;
  void *const ptr = nullptr;
};

#endif // LIBC_FUZZING_STRING_PROTECTED_PAGES_H
