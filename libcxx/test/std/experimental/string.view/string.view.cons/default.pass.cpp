//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


// <string_view>

// constexpr basic_string_view () noexcept;

#include <experimental/string_view>
#include <cassert>

template<typename T>
void test () {
#if _LIBCPP_STD_VER > 11
    {
    constexpr T sv1;
    static_assert ( sv1.size() == 0, "" );
    static_assert ( sv1.empty(), "");
    }
#endif

    {
    T sv1;
    assert ( sv1.size() == 0 );
    assert ( sv1.empty());
    }
}

int main () {
    typedef std::experimental::string_view    string_view;
    typedef std::experimental::u16string_view u16string_view;
    typedef std::experimental::u32string_view u32string_view;
    typedef std::experimental::wstring_view   wstring_view;

    test<string_view> ();
    test<u16string_view> ();
    test<u32string_view> ();
    test<wstring_view> ();

}
