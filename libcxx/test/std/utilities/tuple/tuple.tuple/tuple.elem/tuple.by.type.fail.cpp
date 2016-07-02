//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

#include <tuple>
#include <string>

struct UserType {};

void test_bad_index() {
    std::tuple<long, long, char, std::string, char, UserType, char> t1;
    (void)std::get<int>(t1); // expected-error@tuple:* {{type not found}}
    (void)std::get<long>(t1); // expected-note {{requested here}}
    (void)std::get<char>(t1); // expected-note {{requested here}}
        // expected-error@tuple:* 2 {{type occurs more than once}}
}

void test_bad_return_type() {
    typedef std::unique_ptr<int> upint;
    std::tuple<upint> t;
    upint p = std::get<upint>(t); // expected-error{{deleted copy constructor}}
}

int main()
{
    test_bad_index();
    test_bad_return_type();
}
