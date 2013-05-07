//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <cwctype>

#include <cwctype>
#include <type_traits>

#ifndef WEOF
#error WEOF not defined
#endif

#ifdef iswalnum
#error iswalnum defined
#endif

#ifdef iswalpha
#error iswalpha defined
#endif

#ifdef iswblank
#error iswblank defined
#endif

#ifdef iswcntrl
#error iswcntrl defined
#endif

#ifdef iswdigit
#error iswdigit defined
#endif

#ifdef iswgraph
#error iswgraph defined
#endif

#ifdef iswlower
#error iswlower defined
#endif

#ifdef iswprint
#error iswprint defined
#endif

#ifdef iswpunct
#error iswpunct defined
#endif

#ifdef iswspace
#error iswspace defined
#endif

#ifdef iswupper
#error iswupper defined
#endif

#ifdef iswxdigit
#error iswxdigit defined
#endif

#ifdef iswctype
#error iswctype defined
#endif

#ifdef wctype
#error wctype defined
#endif

#ifdef towlower
#error towlower defined
#endif

#ifdef towupper
#error towupper defined
#endif

#ifdef towctrans
#error towctrans defined
#endif

#ifdef wctrans
#error wctrans defined
#endif

int main()
{
    std::wint_t w = 0;
    std::wctrans_t wctr = 0;
    std::wctype_t wct = 0;
    static_assert((std::is_same<decltype(std::iswalnum(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswalpha(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswblank(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswcntrl(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswdigit(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswgraph(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswlower(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswprint(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswpunct(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswspace(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswupper(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswxdigit(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::iswctype(w, wct)), int>::value), "");
    static_assert((std::is_same<decltype(std::wctype("")), std::wctype_t>::value), "");
    static_assert((std::is_same<decltype(std::towlower(w)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::towupper(w)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::towctrans(w, wctr)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::wctrans("")), std::wctrans_t>::value), "");
}
