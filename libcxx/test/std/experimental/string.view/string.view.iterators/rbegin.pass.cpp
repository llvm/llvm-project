//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <string_view>

// const_iterator rbegin() const;

#include <experimental/string_view>
#include <cassert>

#include "test_macros.h"

template <class S>
void
test(S s)
{
    const S& cs = s;
    typename S::reverse_iterator b = s.rbegin();
    typename S::const_reverse_iterator cb1 = cs.rbegin();
    typename S::const_reverse_iterator cb2 = s.crbegin();
    if (!s.empty())
    {
        const size_t last = s.size() - 1;
        assert(   *b ==  s[last]);
        assert(  &*b == &s[last]);
        assert( *cb1 ==  s[last]);
        assert(&*cb1 == &s[last]);
        assert( *cb2 ==  s[last]);
        assert(&*cb2 == &s[last]);

    }
    assert(  b == cb1);
    assert(  b == cb2);
    assert(cb1 == cb2);
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
