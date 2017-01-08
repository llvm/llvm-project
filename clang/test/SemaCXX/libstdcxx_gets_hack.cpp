// RUN: %clang_cc1 -fsyntax-only %s -std=c++14 -verify

// This is a test for an egregious hack in Clang that works around
// an issue with libstdc++'s detection of whether glibc provides a
// ::gets function. If there is no ::gets, ignore
//   using ::gets;
// in namespace std.
//
// See PR18402 and gcc.gnu.org/PR77795 for more details.

#ifdef BE_THE_HEADER

#pragma GCC system_header
namespace std {
  using ::gets;
  using ::getx; // expected-error {{no member named 'getx'}}
}

#else

#define BE_THE_HEADER
#include "libstdcxx_pointer_return_false_hack.cpp"

namespace foo {
  using ::gets; // expected-error {{no member named 'gets'}}
}

#endif
