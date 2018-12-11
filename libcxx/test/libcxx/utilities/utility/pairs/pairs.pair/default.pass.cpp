//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// constexpr pair();


#include <utility>
#include <type_traits>

#include "archetypes.hpp"


int main() {
    using NonThrowingDefault = NonThrowingTypes::DefaultOnly;
    using ThrowingDefault = NonTrivialTypes::DefaultOnly;
    static_assert(!std::is_nothrow_default_constructible<std::pair<ThrowingDefault, ThrowingDefault>>::value, "");
    static_assert(!std::is_nothrow_default_constructible<std::pair<NonThrowingDefault, ThrowingDefault>>::value, "");
    static_assert(!std::is_nothrow_default_constructible<std::pair<ThrowingDefault, NonThrowingDefault>>::value, "");
    static_assert( std::is_nothrow_default_constructible<std::pair<NonThrowingDefault, NonThrowingDefault>>::value, "");
}
