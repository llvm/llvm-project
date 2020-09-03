//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <unordered_set>
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-deduction-guides
// XFAIL: clang-6, apple-clang-9.0, apple-clang-9.1, apple-clang-10.0.0

// template<class InputIterator,
//        class Hash = hash<iter-value-type<InputIterator>>,
//        class Pred = equal_to<iter-value-type<InputIterator>>,
//        class Allocator = allocator<iter-value-type<InputIterator>>>
// unordered_multiset(InputIterator, InputIterator, typename see below::size_type = see below,
//                    Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_multiset<iter-value-type<InputIterator>,
//                         Hash, Pred, Allocator>;
//
// template<class T, class Hash = hash<T>,
//        class Pred = equal_to<T>, class Allocator = allocator<T>>
// unordered_multiset(initializer_list<T>, typename see below::size_type = see below,
//                    Hash = Hash(), Pred = Pred(), Allocator = Allocator())
//   -> unordered_multiset<T, Hash, Pred, Allocator>;
//
// template<class InputIterator, class Allocator>
// unordered_multiset(InputIterator, InputIterator, typename see below::size_type, Allocator)
//   -> unordered_multiset<iter-value-type<InputIterator>,
//                         hash<iter-value-type<InputIterator>>,
//                         equal_to<iter-value-type<InputIterator>>,
//                         Allocator>;
//
// template<class InputIterator, class Hash, class Allocator>
// unordered_multiset(InputIterator, InputIterator, typename see below::size_type,
//                    Hash, Allocator)
//   -> unordered_multiset<iter-value-type<InputIterator>, Hash,
//                         equal_to<iter-value-type<InputIterator>>,
//                         Allocator>;
//
// template<class T, class Allocator>
// unordered_multiset(initializer_list<T>, typename see below::size_type, Allocator)
//   -> unordered_multiset<T, hash<T>, equal_to<T>, Allocator>;
//
// template<class T, class Hash, class Allocator>
// unordered_multiset(initializer_list<T>, typename see below::size_type, Hash, Allocator)
//   -> unordered_multiset<T, Hash, equal_to<T>, Allocator>;

#include <functional>
#include <unordered_set>

int main(int, char**)
{
    {
        // cannot deduce Key from nothing
        std::unordered_multiset s;
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }
    {
        // cannot deduce Key from just (Size)
        std::unordered_multiset s(42);
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }
    {
        // cannot deduce Key from just (Size, Hash)
        std::unordered_multiset s(42, std::hash<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Pred)
        std::unordered_multiset s(42, std::hash<int>(), std::equal_to<>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Pred, Allocator)
        std::unordered_multiset s(42, std::hash<int>(), std::equal_to<>(), std::allocator<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }
    {
        // cannot deduce Key from just (Allocator)
        std::unordered_multiset s(std::allocator<int>{});
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }
    {
        // cannot deduce Key from just (Size, Allocator)
        std::unordered_multiset s(42, std::allocator<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }
    {
        // cannot deduce Key from just (Size, Hash, Allocator)
        std::unordered_multiset s(42, std::hash<short>(), std::allocator<int>());
            // expected-error@-1{{no viable constructor or deduction guide for deduction of template arguments of 'unordered_multiset'}}
    }

    return 0;
}
