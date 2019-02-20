//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// <optional>

// T shall be an object type and shall satisfy the requirements of Destructible

#include <optional>

using std::optional;

struct X
{
private:
    ~X() {}
};

int main(int, char**)
{
    using std::optional;
    {
        // expected-error-re@optional:* 2 {{static_assert failed{{.*}} "instantiation of optional with a reference type is ill-formed}}
        optional<int&> opt1;
        optional<int&&> opt2;
    }
    {
        // expected-error-re@optional:* {{static_assert failed{{.*}} "instantiation of optional with a non-destructible type is ill-formed"}}
        optional<X> opt3;
    }
    {
        // expected-error-re@optional:* {{static_assert failed{{.*}} "instantiation of optional with a non-object type is undefined behavior"}}
        // expected-error-re@optional:* {{static_assert failed{{.*}} "instantiation of optional with a non-destructible type is ill-formed}}
        optional<void()> opt4;
    }
    {
        // expected-error-re@optional:* {{static_assert failed{{.*}} "instantiation of optional with a non-object type is undefined behavior"}}
        // expected-error-re@optional:* {{static_assert failed{{.*}} "instantiation of optional with a non-destructible type is ill-formed}}
        // expected-error@optional:* 1+ {{cannot form a reference to 'void'}}
        optional<const void> opt4;
    }
    // FIXME these are garbage diagnostics that Clang should not produce
    // expected-error@optional:* 0+ {{is not a base class}}

  return 0;
}
