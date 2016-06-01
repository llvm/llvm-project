//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// <optional>

// class bad_optional_access : public logic_error

#include <experimental/optional>
#include <type_traits>

int main()
{
    using std::experimental::bad_optional_access;

    static_assert(std::is_base_of<std::logic_error, bad_optional_access>::value, "");
    static_assert(std::is_convertible<bad_optional_access*, std::logic_error*>::value, "");
}
