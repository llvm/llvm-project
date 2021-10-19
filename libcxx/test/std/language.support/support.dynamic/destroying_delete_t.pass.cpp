//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// struct destroying_delete_t {
//   explicit destroying_delete_t() = default;
// };
// inline constexpr destroying_delete_t destroying_delete{};

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <new>

#include <cassert>
#include "test_macros.h"

struct A {
  void *data;
  A();
  ~A();

  static A* New();
  void operator delete(A*, std::destroying_delete_t);
};

bool A_constructed = false;
bool A_destroyed = false;
bool A_destroying_deleted = false;

A::A() {
  A_constructed = true;
}

A::~A() {
  A_destroyed = true;
}

A* A::New() {
  return new(::operator new(sizeof(A))) A();
}

void A::operator delete(A* a, std::destroying_delete_t) {
  A_destroying_deleted = true;
  ::operator delete(a);
}

// Only test the definition of the library feature-test macro when the compiler
// supports the feature -- otherwise we don't define the library feature-test
// macro.
#if defined(__cpp_impl_destroying_delete)
#  if !defined(__cpp_lib_destroying_delete)
#    error "Expected __cpp_lib_destroying_delete to be defined"
#  elif __cpp_lib_destroying_delete < 201806L
#    error "Unexpected value of __cpp_lib_destroying_delete"
#  endif
#else
#  if defined(__cpp_lib_destroying_delete)
#    error "The library feature-test macro for destroying delete shouldn't be defined when the compiler doesn't support the language feature"
#  endif
#endif

int main(int, char**) {
  // Ensure that we call the destroying delete and not the destructor.
  A* ap = A::New();
  assert(A_constructed);
  delete ap;
  assert(!A_destroyed);
  assert(A_destroying_deleted);
  return 0;
}
