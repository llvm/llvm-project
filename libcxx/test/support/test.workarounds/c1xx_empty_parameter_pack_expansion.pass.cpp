//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// Verify TEST_WORKAROUND_C1XX_EMPTY_PARAMETER_PACK_EXPANSION.

#include <type_traits>

#include "test_workarounds.h"

template<class T>
struct identity {
    using type = T;
};

template<class...> struct list {};

// C1XX believes this function template is not viable when LArgs is an empty
// parameter pack.
template <class ...LArgs>
int f2(typename identity<LArgs>::type..., int i) {
    return i;
}

#ifdef TEST_WORKAROUND_C1XX_EMPTY_PARAMETER_PACK_EXPANSION
// C1XX believes this function template *is* viable when LArgs is an empty
// parameter pack. Conforming compilers believe the two overloads are
// ambiguous when LArgs is an empty pack.
template <class ...LArgs>
int f2(int i) {
    return i;
}
#endif

template <class ...LArgs, class ...Args>
int f1(list<LArgs...>, Args&&... args) {
    return f2<LArgs const&...>(args...);
}

int main() {
    f1(list<>{}, 42);
}
