//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <experimental/any>

// template <class ValueType>
// ValueType const any_cast(any const&);
//
// template <class ValueType>
// ValueType any_cast(any &);
//
// template <class ValueType>
// ValueType any_cast(any &&);

// Test instantiating the any_cast with a non-copyable type.

#include <experimental/any>

using std::experimental::any;
using std::experimental::any_cast;

struct no_copy
{
    no_copy() {}
    no_copy(no_copy &&) {}
private:
    no_copy(no_copy const &);
};

int main() {
    any a;
    any_cast<no_copy>(static_cast<any&>(a));
    any_cast<no_copy>(static_cast<any const&>(a));
    any_cast<no_copy>(static_cast<any &&>(a));
    // expected-error@experimental/any:* 3 {{static_assert failed "_ValueType is required to be a reference or a CopyConstructible type."}}
    // expected-error@experimental/any:* 3 {{calling a private constructor of class 'no_copy'}}
}
