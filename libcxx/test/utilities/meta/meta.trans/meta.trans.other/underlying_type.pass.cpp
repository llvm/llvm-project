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

int main()
{
    enum E { V = INT_MIN };
    enum F { W = UINT_MAX };

    static_assert((std::is_same<std::underlying_type<E>::type, int>::value),
                  "E has the wrong underlying type");
    static_assert((std::is_same<std::underlying_type<F>::type, unsigned>::value),
                  "F has the wrong underlying type");

#if __has_feature(cxx_strong_enums)
    enum G : char { };

    static_assert((std::is_same<std::underlying_type<G>::type, char>::value),
                  "G has the wrong underlying type");
#endif // __has_feature(cxx_strong_enums)
}
