//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr const_iterator end() const;

#include <experimental/string_view>
#include <cassert>

#include "test_macros.h"

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::iterator e = s.end();
    typename S::const_iterator ce1 = cs.end();
    typename S::const_iterator ce2 = s.cend();

    if (s.empty())
    {
        assert(  e ==  s.begin());
        assert(ce1 == cs.begin());
        assert(ce2 ==  s.begin());
    }
    else
    {
        assert(  e !=  s.begin());
        assert(ce1 != cs.begin());
        assert(ce2 !=  s.begin());
    }

    assert(  e -  s.begin() == s.size());
    assert(ce1 - cs.begin() == cs.size());
    assert(ce2 - s.cbegin() == s.size());

    assert(  e == ce1);
    assert(  e == ce2);
    assert(ce1 == ce2);
}


int main()
{
    typedef std::experimental::string_view    string_view;
    typedef std::experimental::u16string_view u16string_view;
    typedef std::experimental::u32string_view u32string_view;
    typedef std::experimental::wstring_view   wstring_view;

    test(string_view   ());
    test(u16string_view());
    test(u32string_view());
    test(wstring_view  ());
    test(string_view   ( "123"));
    test(wstring_view  (L"123"));
#if TEST_STD_VER >= 11
    test(u16string_view{u"123"});
    test(u32string_view{U"123"});
#endif

#if TEST_STD_VER > 11
    {
    constexpr string_view       sv { "123", 3 };
    constexpr u16string_view u16sv {u"123", 3 };
    constexpr u32string_view u32sv {U"123", 3 };
    constexpr wstring_view     wsv {L"123", 3 };

    static_assert (    sv.begin() !=    sv.end(), "" );
    static_assert ( u16sv.begin() != u16sv.end(), "" );
    static_assert ( u32sv.begin() != u32sv.end(), "" );
    static_assert (   wsv.begin() !=   wsv.end(), "" );

    static_assert (    sv.begin() !=    sv.cend(), "" );
    static_assert ( u16sv.begin() != u16sv.cend(), "" );
    static_assert ( u32sv.begin() != u32sv.cend(), "" );
    static_assert (   wsv.begin() !=   wsv.cend(), "" );
    }
#endif
}
