//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// underlying_type

#include <type_traits>
#include <climits>

#include "test_macros.h"

enum E { V = INT_MIN };
enum F { W = UINT_MAX };

int main()
{
#if !defined(_WIN32) || defined(__MINGW32__)
    typedef unsigned ExpectUnsigned;
#else
    typedef int ExpectUnsigned; // MSVC's ABI doesn't follow the Standard
#endif
    static_assert((std::is_same<std::underlying_type<E>::type, int>::value),
                  "E has the wrong underlying type");
    static_assert((std::is_same<std::underlying_type<F>::type, ExpectUnsigned>::value),
                  "F has the wrong underlying type");

#if TEST_STD_VER > 11
    static_assert((std::is_same<std::underlying_type_t<E>, int>::value), "");
    static_assert((std::is_same<std::underlying_type_t<F>, ExpectUnsigned>::value), "");
#endif

#if TEST_STD_VER >= 11
    enum G : char { };

    static_assert((std::is_same<std::underlying_type<G>::type, char>::value),
                  "G has the wrong underlying type");
#if TEST_STD_VER > 11
    static_assert((std::is_same<std::underlying_type_t<G>, char>::value), "");
#endif
#endif // TEST_STD_VER >= 11
}
