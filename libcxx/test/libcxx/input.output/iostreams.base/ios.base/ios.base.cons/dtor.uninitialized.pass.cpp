//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-exceptions

// The fix for issue 57964 requires an updated dylib due to explicit
// instantiations. That means Apple backdeployment targets remain broken.
// XFAIL: using-built-library-before-llvm-19

// <ios>

// class ios_base

// ~ios_base()
//
// Destroying a constructed ios_base object that has not been
// initialized by basic_ios::init is undefined behaviour. This can
// happen in practice, make sure the undefined behaviour is handled
// gracefully.
//
//
// [ios.base.cons]/1
//
// ios_base();
// Effects: Each ios_base member has an indeterminate value after construction.
// The object's members shall be initialized by calling basic_ios::init before
// the object's first use or before it is destroyed, whichever comes first;
// otherwise the behavior is undefined.
//
// [basic.ios.cons]/2
//
// basic_ios();
// Effects: Leaves its member objects uninitialized.  The object shall be
// initialized by calling basic_ios::init before its first use or before it is
// destroyed, whichever comes first; otherwise the behavior is undefined.
//
// ostream and friends have a basic_ios virtual base.
// [class.base.init]/13
// In a non-delegating constructor, initialization proceeds in the
// following order:
// - First, and only for the constructor of the most derived class
//   ([intro.object]), virtual base classes are initialized ...
//
// So in this example
// struct Foo : AlwaysThrows, std::ostream {
//    Foo() : AlwaysThrows{}, std::ostream{nullptr} {}
// };
//
// Here
// - the ios_base object is constructed
// - the AlwaysThrows object is constructed and throws an exception
// - the AlwaysThrows object is destrodyed
// - the ios_base object is destroyed
//
// The ios_base object is destroyed before it has been initialized and runs
// into undefined behavior. By using __loc_ as a sentinel we can avoid
// accessing uninitialized memory in the destructor.

#include <ostream>

struct AlwaysThrows {
  AlwaysThrows() { throw 1; }
};

struct Foo : AlwaysThrows, std::ostream {
  Foo() : AlwaysThrows(), std::ostream(nullptr) {}
};

int main(int, char**) {
  try {
    Foo foo;
  } catch (...) {
  };
  return 0;
}
