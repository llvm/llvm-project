//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string_view>

// constexpr const_iterator rend() const;

#include <experimental/string_view>
#include <cassert>

#include "test_macros.h"

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::reverse_iterator e = s.rend();
    typename S::const_reverse_iterator ce1 = cs.rend();
    typename S::const_reverse_iterator ce2 = s.crend();

    if (s.empty())
    {
        assert(  e ==  s.rbegin());
        assert(ce1 == cs.rbegin());
        assert(ce2 ==  s.rbegin());
    }
    else
    {
        assert(  e !=  s.rbegin());
        assert(ce1 != cs.rbegin());
        assert(ce2 !=  s.rbegin());
    }

    assert(  e -  s.rbegin() == s.size());
    assert(ce1 - cs.rbegin() == cs.size());
    assert(ce2 - s.crbegin() == s.size());

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
}
