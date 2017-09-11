//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <system_error>
// class error_code

// Make sure that the error_code bits of <system_error> are self-contained.

#include <system_error>

int main()
{
    std::error_code x;
    (void) x.category();   // returns a std::error_category &
    (void) x.default_error_condition(); // std::error_condition
    (void) x.message();    // returns a std::string
}
