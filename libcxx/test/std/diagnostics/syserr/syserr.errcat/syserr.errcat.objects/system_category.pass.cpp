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

// const error_category& system_category();

#include <system_error>
#include <cassert>
#include <string>
#include <cerrno>


void test_message_leaves_errno_unchanged() {
    errno = E2BIG; // something that message will never generate
    const std::error_category& e_cat1 = std::system_category();
    e_cat1.message(-1);
    assert(errno == E2BIG);
}

int main()
{
    const std::error_category& e_cat1 = std::system_category();
    std::error_condition e_cond = e_cat1.default_error_condition(5);
    assert(e_cond.value() == 5);
    assert(e_cond.category() == std::generic_category());
    e_cond = e_cat1.default_error_condition(5000);
    assert(e_cond.value() == 5000);
    assert(e_cond.category() == std::system_category());
    {
        test_message_leaves_errno_unchanged();
    }
}
