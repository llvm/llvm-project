//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <cctype>

#include <cctype>
#include <type_traits>
#include <cassert>

#ifdef isalnum
#error isalnum defined
#endif

#ifdef isalpha
#error isalpha defined
#endif

#ifdef isblank
#error isblank defined
#endif

#ifdef iscntrl
#error iscntrl defined
#endif

#ifdef isdigit
#error isdigit defined
#endif

#ifdef isgraph
#error isgraph defined
#endif

#ifdef islower
#error islower defined
#endif

#ifdef isprint
#error isprint defined
#endif

#ifdef ispunct
#error ispunct defined
#endif

#ifdef isspace
#error isspace defined
#endif

#ifdef isupper
#error isupper defined
#endif

#ifdef isxdigit
#error isxdigit defined
#endif

#ifdef tolower
#error tolower defined
#endif

#ifdef toupper
#error toupper defined
#endif

int main()
{
    static_assert((std::is_same<decltype(std::isalnum(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isalpha(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isblank(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::iscntrl(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isdigit(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isgraph(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::islower(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isprint(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::ispunct(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isspace(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isupper(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::isxdigit(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::tolower(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::toupper(0)), int>::value), "");

    assert(isalnum('a'));
    assert(isalpha('a'));
    assert(isblank(' '));
    assert(!iscntrl(' '));
    assert(!isdigit('a'));
    assert(isgraph('a'));
    assert(islower('a'));
    assert(isprint('a'));
    assert(!ispunct('a'));
    assert(!isspace('a'));
    assert(!isupper('a'));
    assert(isxdigit('a'));
    assert(tolower('A') == 'a');
    assert(toupper('a') == 'A');
}
