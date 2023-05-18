// RUN: %clang_cc1 -fsyntax-only %s -std=c++11 -verify

// This is a test for an egregious hack in Clang that works around
// an issue with libstdc++-4.2's <tr1/hashtable> implementation.
// The code in question returns 'false' from a function with a pointer
// return type, which is ill-formed in C++11.

#ifdef BE_THE_HEADER

#pragma GCC system_header
namespace std {
  namespace tr1 {
    template<typename T> struct hashnode;
    template<typename T> struct hashtable {
      typedef hashnode<T> node;
      node *find_node() {
        // This is ill-formed in C++11, per core issue 903, but we accept
        // it anyway in a system header.
        return false;
      }
    };
  }
}

#else

#define BE_THE_HEADER
#include "libstdcxx_pointer_return_false_hack.cpp"

auto *test1 = std::tr1::hashtable<int>().find_node();

void *test2() { return false; } // expected-error {{cannot initialize}}

#endif
