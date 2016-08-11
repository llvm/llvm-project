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
// ValueType const* any_cast(any const *) noexcept;
//
// template <class ValueType>
// ValueType * any_cast(any *) noexcept;

#include <any>

using std::any;
using std::any_cast;

int main()
{
    any a(1);
    any_cast<int &>(&a); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
    any_cast<int &&>(&a); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
    any_cast<int const &>(&a); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
    any_cast<int const&&>(&a); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
    any const& a2 = a;
    any_cast<int &>(&a2); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
    any_cast<int &&>(&a2); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
    any_cast<int const &>(&a2); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
    any_cast<int const &&>(&a2); // expected-error@any:* 1 {{static_assert failed "_ValueType may not be a reference."}}
}
