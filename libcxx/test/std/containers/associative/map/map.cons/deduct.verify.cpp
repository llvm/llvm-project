//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <map>

// template<class InputIterator,
//          class Compare = less<iter-value-type<InputIterator>>,
//          class Allocator = allocator<iter-value-type<InputIterator>>>
// map(InputIterator, InputIterator,
//          Compare = Compare(), Allocator = Allocator())
//   -> map<iter-value-type<InputIterator>, Compare, Allocator>;
// template<class Key, class Compare = less<Key>, class Allocator = allocator<Key>>
// map(initializer_list<Key>, Compare = Compare(), Allocator = Allocator())
//   -> map<Key, Compare, Allocator>;
// template<class InputIterator, class Allocator>
// map(InputIterator, InputIterator, Allocator)
//   -> map<iter-value-type<InputIterator>, less<iter-value-type<InputIterator>>, Allocator>;
// template<class Key, class Allocator>
// map(initializer_list<Key>, Allocator)
//   -> map<Key, less<Key>, Allocator>;

#include <climits> // INT_MAX
#include <functional>
#include <map>
#include <type_traits>

struct NotAnAllocator {
    friend bool operator<(NotAnAllocator, NotAnAllocator) { return false; }
};

using P = std::pair<int, long>;
using PC = std::pair<const int, long>;

int main(int, char**)
{
    {
        // cannot deduce Key and T from nothing
        std::map m; // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // cannot deduce Key and T from just (Compare)
        std::map m(std::less<int>{});
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // cannot deduce Key and T from just (Compare, Allocator)
        std::map m(std::less<int>{}, std::allocator<PC>{});
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // cannot deduce Key and T from just (Allocator)
        std::map m(std::allocator<PC>{});
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // refuse to rebind the allocator if Allocator::value_type is not exactly what we expect
        const P arr[] = { {1,1L}, {2,2L}, {3,3L} };
        std::map m(arr, arr + 3, std::allocator<P>());
            // expected-error-re@map:* {{static assertion failed{{( due to requirement '.*')?}}{{.*}}Allocator::value_type must be same type as value_type}}
    }
    {
        // cannot convert from some arbitrary unrelated type
        NotAnAllocator a;
        std::map m(a); // expected-error{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // cannot deduce that the inner braced things should be std::pair and not something else
        std::map m{ {1,1L}, {2,2L}, {3,3L} };
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // cannot deduce that the inner braced things should be std::pair and not something else
        std::map m({ {1,1L}, {2,2L}, {3,3L} }, std::less<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // cannot deduce that the inner braced things should be std::pair and not something else
        std::map m({ {1,1L}, {2,2L}, {3,3L} }, std::less<int>(), std::allocator<PC>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // cannot deduce that the inner braced things should be std::pair and not something else
        std::map m({ {1,1L}, {2,2L}, {3,3L} }, std::allocator<PC>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // since we have parens, not braces, this deliberately does not find the initializer_list constructor
        std::map m(P{1,1L});
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }
    {
        // since we have parens, not braces, this deliberately does not find the initializer_list constructor
        std::map m(PC{1,1L});
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'map'}}
    }

    return 0;
}
