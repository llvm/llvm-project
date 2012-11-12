//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// test <cstdlib>

#include <cstdlib>
#include <type_traits>

#ifndef EXIT_FAILURE
#error EXIT_FAILURE not defined
#endif

#ifndef EXIT_SUCCESS
#error EXIT_SUCCESS not defined
#endif

#ifndef MB_CUR_MAX
#error MB_CUR_MAX not defined
#endif

#ifndef NULL
#error NULL not defined
#endif

#ifndef RAND_MAX
#error RAND_MAX not defined
#endif

int main()
{
    std::size_t s = 0;
    std::div_t d;
    std::ldiv_t ld;
    std::lldiv_t lld;
    char** endptr = 0;
    static_assert((std::is_same<decltype(std::atof("")), double>::value), "");
    static_assert((std::is_same<decltype(std::atoi("")), int>::value), "");
    static_assert((std::is_same<decltype(std::atol("")), long>::value), "");
    static_assert((std::is_same<decltype(std::atoll("")), long long>::value), "");
    static_assert((std::is_same<decltype(std::getenv("")), char*>::value), "");
    static_assert((std::is_same<decltype(std::strtod("", endptr)), double>::value), "");
    static_assert((std::is_same<decltype(std::strtof("", endptr)), float>::value), "");
    static_assert((std::is_same<decltype(std::strtold("", endptr)), long double>::value), "");
    static_assert((std::is_same<decltype(std::strtol("", endptr,0)), long>::value), "");
    static_assert((std::is_same<decltype(std::strtoll("", endptr,0)), long long>::value), "");
    static_assert((std::is_same<decltype(std::strtoul("", endptr,0)), unsigned long>::value), "");
    static_assert((std::is_same<decltype(std::strtoull("", endptr,0)), unsigned long long>::value), "");
    static_assert((std::is_same<decltype(std::rand()), int>::value), "");
    static_assert((std::is_same<decltype(std::srand(0)), void>::value), "");
    static_assert((std::is_same<decltype(std::calloc(0,0)), void*>::value), "");
    static_assert((std::is_same<decltype(std::free(0)), void>::value), "");
    static_assert((std::is_same<decltype(std::malloc(0)), void*>::value), "");
    static_assert((std::is_same<decltype(std::realloc(0,0)), void*>::value), "");
    static_assert((std::is_same<decltype(std::abort()), void>::value), "");
    static_assert((std::is_same<decltype(std::atexit(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::exit(0)), void>::value), "");
    static_assert((std::is_same<decltype(std::_Exit(0)), void>::value), "");
    static_assert((std::is_same<decltype(std::getenv("")), char*>::value), "");
    static_assert((std::is_same<decltype(std::system("")), int>::value), "");
    static_assert((std::is_same<decltype(std::bsearch(0,0,0,0,0)), void*>::value), "");
    static_assert((std::is_same<decltype(std::qsort(0,0,0,0)), void>::value), "");
    static_assert((std::is_same<decltype(std::abs(0)), int>::value), "");
    static_assert((std::is_same<decltype(std::abs((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(std::abs((long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(std::labs((long)0)), long>::value), "");
    static_assert((std::is_same<decltype(std::llabs((long long)0)), long long>::value), "");
    static_assert((std::is_same<decltype(std::div(0,0)), std::div_t>::value), "");
    static_assert((std::is_same<decltype(std::div(0L,0L)), std::ldiv_t>::value), "");
    static_assert((std::is_same<decltype(std::div(0LL,0LL)), std::lldiv_t>::value), "");
    static_assert((std::is_same<decltype(std::ldiv(0L,0L)), std::ldiv_t>::value), "");
    static_assert((std::is_same<decltype(std::lldiv(0LL,0LL)), std::lldiv_t>::value), "");
    static_assert((std::is_same<decltype(std::mblen("",0)), int>::value), "");
    wchar_t* pw = 0;
    const wchar_t* pwc = 0;
    char* pc = 0;
    static_assert((std::is_same<decltype(std::mbtowc(pw,"",0)), int>::value), "");
    static_assert((std::is_same<decltype(std::wctomb(pc,L' ')), int>::value), "");
    static_assert((std::is_same<decltype(std::mbstowcs(pw,"",0)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcstombs(pc,pwc,0)), std::size_t>::value), "");
}
