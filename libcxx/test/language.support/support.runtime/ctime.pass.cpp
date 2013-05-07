//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test <ctime>

#include <ctime>
#include <type_traits>

#ifndef NULL
#error NULL not defined
#endif

#ifndef CLOCKS_PER_SEC
#error CLOCKS_PER_SEC not defined
#endif

int main()
{
    std::clock_t c = 0;
    std::size_t s = 0;
    std::time_t t = 0;
    std::tm tm = {0};
    static_assert((std::is_same<decltype(std::clock()), std::clock_t>::value), "");
    static_assert((std::is_same<decltype(std::difftime(t,t)), double>::value), "");
    static_assert((std::is_same<decltype(std::mktime(&tm)), std::time_t>::value), "");
    static_assert((std::is_same<decltype(std::time(&t)), std::time_t>::value), "");
    static_assert((std::is_same<decltype(std::asctime(&tm)), char*>::value), "");
    static_assert((std::is_same<decltype(std::ctime(&t)), char*>::value), "");
    static_assert((std::is_same<decltype(std::gmtime(&t)), std::tm*>::value), "");
    static_assert((std::is_same<decltype(std::localtime(&t)), std::tm*>::value), "");
    char* c1 = 0;
    const char* c2 = 0;
    static_assert((std::is_same<decltype(std::strftime(c1,s,c2,&tm)), std::size_t>::value), "");
}
