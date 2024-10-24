//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Make sure that we can form unique_ptrs to incomplete types and perform restricted
// operations on them. This requires setting up a TU where the type is complete and
// the unique_ptr is created and destroyed, and a TU where the type is incomplete and
// we check that a restricted set of operations can be performed on the unique_ptr.

// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.tu1.o -DCOMPLETE
// RUN: %{cxx} %s %{flags} %{compile_flags} -c -o %t.tu2.o -DINCOMPLETE
// RUN: %{cxx} %t.tu1.o %t.tu2.o %{flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include <memory>
#include <cassert>

struct T;
extern void use(std::unique_ptr<T>& ptr);
extern void use(std::unique_ptr<T[]>& ptr);

#ifdef INCOMPLETE

void use(std::unique_ptr<T>& ptr) {
  {
    T* x = ptr.get();
    assert(x != nullptr);
  }
  {
    T& ref = *ptr;
    assert(&ref == ptr.get());
  }
  {
    bool engaged = static_cast<bool>(ptr);
    assert(engaged);
  }
  {
    assert(ptr == ptr);
    assert(!(ptr != ptr));
    assert(!(ptr < ptr));
    assert(!(ptr > ptr));
    assert(ptr <= ptr);
    assert(ptr >= ptr);
  }
}

void use(std::unique_ptr<T[]>& ptr) {
  {
    T* x = ptr.get();
    assert(x != nullptr);
  }
  {
    bool engaged = static_cast<bool>(ptr);
    assert(engaged);
  }
  {
    assert(ptr == ptr);
    assert(!(ptr != ptr));
    assert(!(ptr < ptr));
    assert(!(ptr > ptr));
    assert(ptr <= ptr);
    assert(ptr >= ptr);
  }
}

#endif // INCOMPLETE

#ifdef COMPLETE

struct T {}; // complete the type

int main(int, char**) {
  {
    std::unique_ptr<T> ptr(new T());
    use(ptr);
  }

  {
    std::unique_ptr<T[]> ptr(new T[3]());
    use(ptr);
  }
  return 0;
}

#endif // COMPLETE
