//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_UTILITIES_MEMORY_UTIL_SMARTPTR_SHARED_CONST_TYPES_H
#define TEST_STD_UTILITIES_MEMORY_UTIL_SMARTPTR_SHARED_CONST_TYPES_H

#include <type_traits>

struct bad_ty {};

struct bad_deleter {
  void operator()(bad_ty) {}
};

struct no_move_deleter {
  no_move_deleter(no_move_deleter const&) = delete;
  no_move_deleter(no_move_deleter&&)      = delete;
  void operator()(int*) {}
};

static_assert(!std::is_move_constructible<no_move_deleter>::value, "");

struct no_nullptr_deleter {
  void operator()(int*) const {}
  void operator()(std::nullptr_t) const = delete;
};

struct Base {};
struct Derived : Base {};

template <class T>
class MoveDeleter {
  MoveDeleter();
  MoveDeleter(MoveDeleter const&);

public:
  MoveDeleter(MoveDeleter&&) {}

  explicit MoveDeleter(int) {}

  void operator()(T* ptr) { delete ptr; }
};

#endif // TEST_STD_UTILITIES_MEMORY_UTIL_SMARTPTR_SHARED_CONST_TYPES_H
