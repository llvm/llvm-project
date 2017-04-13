//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <memory>

// unique_ptr

// Test unique_ptr default ctor

// default unique_ptr ctor should require default Deleter ctor

#include <memory>
#include "test_macros.h"

class Deleter {
  Deleter() {}

public:
  Deleter(Deleter&) {}
  Deleter& operator=(Deleter&) { return *this; }

  void operator()(void*) const {}
};

int main() {
#if TEST_STD_VER >= 11
  // expected-error@memory:* {{call to implicitly-deleted default constructor}}
  // expected-note@memory:* {{implicitly deleted because base class 'Deleter' has an inaccessible default constructor}}
#else
  // expected-error@memory:* {{base class 'Deleter' has private default constructor}}
#endif
  std::unique_ptr<int[], Deleter> p; // expected-note {{requested here}}
}
