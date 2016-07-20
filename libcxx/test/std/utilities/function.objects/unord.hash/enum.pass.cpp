//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <functional>

// make sure that we can hash enumeration values
// Not very portable

#include "test_macros.h"

#if TEST_STD_VER >= 14

#include <functional>
#include <cassert>
#include <type_traits>
#include <limits>

enum class Colors { red, orange, yellow, green, blue, indigo, violet };
enum class Cardinals { zero, one, two, three, five=5 };
enum class LongColors : short { red, orange, yellow, green, blue, indigo, violet };
enum class ShortColors : long { red, orange, yellow, green, blue, indigo, violet };
enum class EightBitColors : uint8_t { red, orange, yellow, green, blue, indigo, violet };

enum Fruits { apple, pear, grape, mango, cantaloupe };

template <class T>
void
test()
{
    typedef std::hash<T> H;
    static_assert((std::is_same<typename H::argument_type, T>::value), "" );
    static_assert((std::is_same<typename H::result_type, std::size_t>::value), "" );
    typedef typename std::underlying_type<T>::type under_type;

    H h1;
    std::hash<under_type> h2;
    for (int i = 0; i <= 5; ++i)
    {
        T t(static_cast<T> (i));
        if (sizeof(T) <= sizeof(std::size_t))
            assert(h1(t) == h2(static_cast<under_type>(i)));
    }
}

int main()
{
    test<Cardinals>();

    test<Colors>();
    test<ShortColors>();
    test<LongColors>();
    test<EightBitColors>();

    test<Fruits>();
}
#else
int main () {}
#endif
