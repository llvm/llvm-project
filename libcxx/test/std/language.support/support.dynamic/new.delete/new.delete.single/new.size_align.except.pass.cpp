//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions
// UNSUPPORTED: sanitizer-new-delete

#include <new>
#include <cassert>
#include <limits>
#include <cstdlib>

struct construction_key {};
struct my_bad_alloc : std::bad_alloc {
  my_bad_alloc(const my_bad_alloc&) : self(this) { std::abort(); }
  my_bad_alloc(construction_key) : self(this) {}
  const my_bad_alloc* const self;
};

int new_handler_called = 0;

void my_new_handler() {
  ++new_handler_called;
  throw my_bad_alloc(construction_key());
}

int main(int, char**) {
  std::set_new_handler(my_new_handler);
  try {
    void* x = operator new(std::numeric_limits<std::size_t>::max(), static_cast<std::align_val_t>(32));
    (void)x;
    assert(false);
  } catch (my_bad_alloc const& e) {
    assert(new_handler_called == 1);
    assert(e.self == &e);
  } catch (...) {
    assert(false);
  }
  return 0;
}
