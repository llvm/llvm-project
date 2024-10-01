//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <unordered_map>

// template<class InputIterator,
//          class Hash = hash<iter-key-type<InputIterator>>,
//          class Pred = equal_to<iter-key-type<InputIterator>>,
//          class Allocator = allocator<iter-to-alloc-type<InputIterator>>>
// unordered_map(InputIterator, InputIterator, typename see below::size_type = see below,
//               Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>, Hash, Pred,
//                    Allocator>;
//
// template<class Key, class T, class Hash = hash<Key>,
//          class Pred = equal_to<Key>, class Allocator = allocator<pair<const Key, T>>>
// unordered_map(initializer_list<pair<Key, T>>,
//               typename see below::size_type = see below, Hash = Hash(),
//               Pred = Pred(), Allocator = Allocator())
//   -> unordered_map<Key, T, Hash, Pred, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_map(InputIterator, InputIterator, typename see below::size_type, Allocator)
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>,
//                    hash<iter-key-type<InputIterator>>,
//                    equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_map(InputIterator, InputIterator, Allocator)
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>,
//                    hash<iter-key-type<InputIterator>>,
//                    equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class InputIterator, class Hash, class Allocator>
// unordered_map(InputIterator, InputIterator, typename see below::size_type, Hash, Allocator)
//   -> unordered_map<iter-key-type<InputIterator>, iter-mapped-type<InputIterator>, Hash,
//                    equal_to<iter-key-type<InputIterator>>, Allocator>;
//
// template<class Key, class T, class Allocator>
// unordered_map(initializer_list<pair<Key, T>>, typename see below::size_type, Allocator)
//   -> unordered_map<Key, T, hash<Key>, equal_to<Key>, Allocator>;
//
// template<class Key, class T, class Allocator>
// unordered_map(initializer_list<pair<Key, T>>, Allocator)
//   -> unordered_map<Key, T, hash<Key>, equal_to<Key>, Allocator>;
//
// template<class Key, class T, class Hash, class Allocator>
// unordered_map(initializer_list<pair<Key, T>>, typename see below::size_type, Hash,
//               Allocator)
//   -> unordered_map<Key, T, Hash, equal_to<Key>, Allocator>;

#include <functional>
#include <unordered_map>

int main(int, char**)
{
    using P = std::pair<const int, int>;
    {
        // cannot deduce Key from nothing
        std::unordered_map m; // expected-error-re{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }
    {
        // cannot deduce Key from just (Size)
        std::unordered_map m(42); // expected-error-re{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }
    {
        // cannot deduce Key from just (Size, Hash)
        std::unordered_map m(42, std::hash<int>());
            // expected-error-re@-1{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Pred)
        std::unordered_map m(42, std::hash<int>(), std::equal_to<int>());
            // expected-error-re@-1{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Pred, Allocator)
        std::unordered_map m(42, std::hash<int>(), std::equal_to<int>(), std::allocator<P>());
            // expected-error-re@-1{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }
    {
        // cannot deduce Key from just (Allocator)
        std::unordered_map m(std::allocator<P>{});
            // expected-error-re@-1{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }
    {
        // cannot deduce Key from just (Size, Allocator)
        std::unordered_map m(42, std::allocator<P>());
            // expected-error-re@-1{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Allocator)
        std::unordered_map m(42, std::hash<int>(), std::allocator<P>());
            // expected-error-re@-1{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}unordered_map'}}
    }

  return 0;
}
