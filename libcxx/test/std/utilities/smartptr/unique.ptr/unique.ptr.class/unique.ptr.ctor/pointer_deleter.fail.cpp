//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// Without rvalue references it is impossible to detect when a rvalue deleter
// is given.
// XFAIL: c++98, c++03

// <memory>

// unique_ptr

// unique_ptr<T, const D&>(pointer, D()) should not compile

#include <memory>

#include "test_workarounds.h"

struct Deleter {
  void operator()(int* p) const { delete p; }
};

int main() {
#if defined(TEST_WORKAROUND_UPCOMING_UNIQUE_PTR_CHANGES)
// expected-error@memory:* {{static_assert failed "rvalue deleter bound to reference"}}
#else
// expected-error@+2 {{call to deleted constructor of 'std::unique_ptr<int, const Deleter &>}}
#endif
  std::unique_ptr<int, const Deleter&> s((int*)nullptr, Deleter());
}
