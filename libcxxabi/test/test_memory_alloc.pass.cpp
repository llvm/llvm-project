//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===// 

#include <new>
#include <cassert>
#include <limits>

int new_handler_called = 0;

void my_new_handler() {
  ++new_handler_called;
  throw std::bad_alloc();
}

int main(int, char**) {
  std::set_new_handler(my_new_handler);
  try {
    void* x = operator new[] (std::numeric_limits<std::size_t>::max());
    (void)x;
    assert(false);
  }
  catch (std::bad_alloc const&) {
    assert(new_handler_called == 1);
  } catch (...) {
    assert(false);
  }
  return 0;
}
