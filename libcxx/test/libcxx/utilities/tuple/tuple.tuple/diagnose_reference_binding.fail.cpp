//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <tuple>

// Test the diagnostics libc++ generates for invalid reference binding.
// Libc++ attempts to diagnose the following cases:
//  * Constructing an lvalue reference from an rvalue.
//  * Constructing an rvalue reference from an lvalue.

#include <tuple>
#include <string>

int main() {
    std::allocator<void> alloc;

    // expected-error-re@tuple:* 4 {{static_assert failed{{.*}} "Attempted to construct a reference element in a tuple with an rvalue"}}

    // bind lvalue to rvalue
    std::tuple<int const&> t(42); // expected-note {{requested here}}
    std::tuple<int const&> t1(std::allocator_arg, alloc, 42); // expected-note {{requested here}}
    // bind rvalue to constructed non-rvalue
    std::tuple<std::string &&> t2("hello"); // expected-note {{requested here}}
    std::tuple<std::string &&> t3(std::allocator_arg, alloc, "hello"); // expected-note {{requested here}}

    // FIXME: The below warnings may get emitted as an error, a warning, or not emitted at all
    // depending on the flags used to compile this test.
  {
    // expected-warning@tuple:* 0+ {{binding reference member '__value_' to a temporary value}}
    // expected-error@tuple:* 0+ {{binding reference member '__value_' to a temporary value}}
  }
}
