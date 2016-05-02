//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_category

// constexpr error_category() noexcept;

#include <system_error>
#include <type_traits>
#include <string>
#include <cassert>

#if _LIBCPP_STD_VER > 11

class test1
    : public std::error_category
{
public:
    constexpr test1() = default;  // won't compile if error_category() is not constexpr
    virtual const char* name() const noexcept {return nullptr;}
    virtual std::string message(int) const {return std::string();}
};

#endif  // _LIBCPP_STD_VER > 11

int main()
{
#if _LIBCPP_STD_VER > 11
    static_assert(std::is_nothrow_default_constructible<test1>::value,
                                 "error_category() must exist and be noexcept");
#endif  // _LIBCPP_STD_VER > 11
}
