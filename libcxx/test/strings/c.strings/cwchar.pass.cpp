//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <cwchar>

#include <cwchar>
#include <type_traits>

#ifndef NULL
#error NULL not defined
#endif

#ifndef WCHAR_MAX
#error WCHAR_MAX not defined
#endif

#ifndef WCHAR_MIN
#error WCHAR_MIN not defined
#endif

#ifndef WEOF
#error WEOF not defined
#endif

int main()
{
    std::mbstate_t mb = {0};
    std::size_t s = 0;
    std::tm *tm = 0;
    std::wint_t w = 0;
    ::FILE* fp = 0;
#ifdef __APPLE__
    __darwin_va_list va;
#else
    __builtin_va_list va;
#endif
    char* ns = 0;
    wchar_t* ws = 0;
    static_assert((std::is_same<decltype(std::fwprintf(fp, L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::fwscanf(fp, L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::swprintf(ws, s, L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::swscanf(L"", L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::vfwprintf(fp, L"", va)), int>::value), "");
    static_assert((std::is_same<decltype(std::vfwscanf(fp, L"", va)), int>::value), "");
    static_assert((std::is_same<decltype(std::vswprintf(ws, s, L"", va)), int>::value), "");
    static_assert((std::is_same<decltype(std::vswscanf(L"", L"", va)), int>::value), "");
    static_assert((std::is_same<decltype(std::vwprintf(L"", va)), int>::value), "");
    static_assert((std::is_same<decltype(std::vwscanf(L"", va)), int>::value), "");
    static_assert((std::is_same<decltype(std::wprintf(L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::wscanf(L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::fgetwc(fp)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::fgetws(ws, 0, fp)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::fputwc(L' ', fp)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::fputws(L"", fp)), int>::value), "");
    static_assert((std::is_same<decltype(std::fwide(fp, 0)), int>::value), "");
    static_assert((std::is_same<decltype(std::getwc(fp)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::getwchar()), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::putwc(L' ', fp)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::putwchar(L' ')), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::ungetwc(L' ', fp)), std::wint_t>::value), "");
    static_assert((std::is_same<decltype(std::wcstod(L"", (wchar_t**)0)), double>::value), "");
    static_assert((std::is_same<decltype(std::wcstof(L"", (wchar_t**)0)), float>::value), "");
    static_assert((std::is_same<decltype(std::wcstold(L"", (wchar_t**)0)), long double>::value), "");
    static_assert((std::is_same<decltype(std::wcstol(L"", (wchar_t**)0, 0)), long>::value), "");
    static_assert((std::is_same<decltype(std::wcstoll(L"", (wchar_t**)0, 0)), long long>::value), "");
    static_assert((std::is_same<decltype(std::wcstoul(L"", (wchar_t**)0, 0)), unsigned long>::value), "");
    static_assert((std::is_same<decltype(std::wcstoull(L"", (wchar_t**)0, 0)), unsigned long long>::value), "");
    static_assert((std::is_same<decltype(std::wcscpy(ws, L"")), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcsncpy(ws, L"", s)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcscat(ws, L"")), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcsncat(ws, L"", s)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcscmp(L"", L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::wcscoll(L"", L"")), int>::value), "");
    static_assert((std::is_same<decltype(std::wcsncmp(L"", L"", s)), int>::value), "");
    static_assert((std::is_same<decltype(std::wcsxfrm(ws, L"", s)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcschr((const wchar_t*)0, L' ')), const wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcschr((wchar_t*)0, L' ')), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcscspn(L"", L"")), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcslen(L"")), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcspbrk((const wchar_t*)0, L"")), const wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcspbrk((wchar_t*)0, L"")), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcsrchr((const wchar_t*)0, L' ')), const wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcsrchr((wchar_t*)0, L' ')), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcsspn(L"", L"")), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcsstr((const wchar_t*)0, L"")), const wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcsstr((wchar_t*)0, L"")), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcstok(ws, L"", (wchar_t**)0)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wmemchr((const wchar_t*)0, L' ', s)), const wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wmemchr((wchar_t*)0, L' ', s)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wmemcmp(L"", L"", s)), int>::value), "");
    static_assert((std::is_same<decltype(std::wmemcpy(ws, L"", s)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wmemmove(ws, L"", s)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wmemset(ws, L' ', s)), wchar_t*>::value), "");
    static_assert((std::is_same<decltype(std::wcsftime(ws, s, L"", tm)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::btowc(0)), wint_t>::value), "");
    static_assert((std::is_same<decltype(std::wctob(w)), int>::value), "");
    static_assert((std::is_same<decltype(std::mbsinit(&mb)), int>::value), "");
    static_assert((std::is_same<decltype(std::mbrlen("", s, &mb)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::mbrtowc(ws, "", s, &mb)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcrtomb(ns, L' ', &mb)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::mbsrtowcs(ws, (const char**)0, s, &mb)), std::size_t>::value), "");
    static_assert((std::is_same<decltype(std::wcsrtombs(ns, (const wchar_t**)0, s, &mb)), std::size_t>::value), "");
}
