//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14

// <any>

// template <class ValueType>
// ValueType const any_cast(any const&);
//
// template <class ValueType>
// ValueType any_cast(any &);
//
// template <class ValueType>
// ValueType any_cast(any &&);

// Test instantiating the any_cast with a non-copyable type.

#include <any>

using std::any;
using std::any_cast;

struct no_copy
{
    no_copy() {}
    no_copy(no_copy &&) {}
private:
    no_copy(no_copy const &);
};

int main() {
    any a;
    any_cast<no_copy>(static_cast<any&>(a)); // expected-note {{requested here}}
    any_cast<no_copy>(static_cast<any const&>(a)); // expected-note {{requested here}}
    any_cast<no_copy>(static_cast<any &&>(a)); // expected-note {{requested here}}
    // expected-error@any:* 3 {{static_assert failed "_ValueType is required to be a reference or a CopyConstructible type."}}
    // expected-error@any:* 2 {{calling a private constructor of class 'no_copy'}}
}