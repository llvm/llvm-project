//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__locale_dir/support/windows.h>
#include <cctype>
#include <charconv>
#include <clocale> // std::localeconv() & friends
#include <cstdarg> // va_start & friends
#include <cstddef>
#include <cstdio>  // std::vsnprintf & friends
#include <cstdlib> // std::strtof & friends
#include <ctime>   // std::strftime
#include <cwchar>  // wide char manipulation
#include <string_view>
#include <windows.h>

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
namespace __locale {

//
// Locale management
//
// FIXME: base and mask currently unused. Needs manual work to construct the new locale
__locale_t __newlocale(int /*mask*/, const char* locale, __locale_t /*base*/) {
  return {::_create_locale(LC_ALL, locale), locale};
}

__lconv_t* __localeconv(__locale_t& loc) {
  __locale_guard __current(loc);
  lconv* lc = std::localeconv();
  if (!lc)
    return lc;
  return loc.__store_lconv(lc);
}

namespace {
const char* __codepage_name(unsigned int __codepage) {
  switch (__codepage) {
  case 0:
    // If no ANSI code page is available, only Unicode can be used for the locale.
    // In this case, the value is CP_ACP (0).
    // Such a locale cannot be set as the system locale.
    // Applications that do not support Unicode do not work correctly with locales
    // marked as "Unicode only".
    return nullptr;
  case 037:
    return "IBM037";
  case 437:
    return "IBM437";
  case 500:
    return "IBM500";
  case 708:
    return "ASMO-708";
  case 709:
    return "ASMO_449";
  case 775:
    return "IBM775";
  case 850:
    return "IBM850";
  case 852:
    return "IBM852";
  case 855:
    return "IBM855";
  case 857:
    return "IBM857";
  case 858:
    return "IBM00858";
  case 860:
    return "IBM860";
  case 861:
    return "IBM861";
  case 862:
    return "IBM862";
  case 863:
    return "IBM863";
  case 864:
    return "IBM864";
  case 865:
    return "IBM865";
  case 866:
    return "IBM866";
  case 869:
    return "IBM869";
  case 870:
    return "IBM870";
  case 874:
    return "windows-874";
  case 932:
    return "Shift_JIS";
  case 936:
    return "GB2312";
  case 949:
    return "KS_C_5601-1989";
  case 950:
    return "Big5";
  case 1026:
    return "IBM1026";
  case 1047:
    return "IBM1047";
  case 1140:
    return "IBM01140";
  case 1141:
    return "IBM01141";
  case 1142:
    return "IBM01142";
  case 1143:
    return "IBM01143";
  case 1144:
    return "IBM01144";
  case 1145:
    return "IBM01145";
  case 1146:
    return "IBM01146";
  case 1147:
    return "IBM01147";
  case 1148:
    return "IBM01148";
  case 1149:
    return "IBM01149";
  case 1200:
    return "UTF-16LE";
  case 1201:
    return "UTF-16BE";
  case 1250:
    return "windows-1250";
  case 1251:
    return "windows-1251";
  case 1252:
    return "windows-1252";
  case 1253:
    return "windows-1253";
  case 1254:
    return "windows-1254";
  case 1255:
    return "windows-1255";
  case 1256:
    return "windows-1256";
  case 1257:
    return "windows-1257";
  case 1258:
    return "windows-1258";
  case 10000:
    return "macintosh";
  case 12000:
    return "UTF-32LE";
  case 12001:
    return "UTF-32BE";
  case 20127:
    return "US-ASCII";
  case 20273:
    return "IBM273";
  case 20277:
    return "IBM277";
  case 20278:
    return "IBM278";
  case 20280:
    return "IBM280";
  case 20284:
    return "IBM284";
  case 20285:
    return "IBM285";
  case 20290:
    return "IBM290";
  case 20297:
    return "IBM297";
  case 20420:
    return "IBM420";
  case 20423:
    return "IBM423";
  case 20424:
    return "IBM424";
  case 20838:
    return "IBM-Thai";
  case 20866:
    return "KOI8-R";
  case 20871:
    return "IBM871";
  case 20880:
    return "IBM880";
  case 20905:
    return "IBM905";
  case 20924:
    return "IBM00924";
  case 20932:
    return "EUC-JP";
  case 21866:
    return "KOI8-U";
  case 28591:
    return "ISO-8859-1";
  case 28592:
    return "ISO-8859-2";
  case 28593:
    return "ISO-8859-3";
  case 28594:
    return "ISO-8859-4";
  case 28595:
    return "ISO-8859-9";
  case 28596:
    return "ISO-8859-10";
  case 28597:
    return "ISO-8859-7";
  case 28598:
    return "ISO-8859-8";
  case 28599:
    return "ISO-8859-9-Windows-Latin-5";
  case 28603:
    return "ISO-8859-13";
  case 28605:
    return "ISO-8859-15";
  case 38598:
    return "ISO-8859-8-I";
  case 50220:
  case 50221:
  case 50222:
    return "ISO-2022-JP";
  case 51932:
    return "EUC-JP";
  case 51936:
    return "GB2312";
  case 51949:
    return "EUC-KR";
  case 52936:
    return "HZ-GB-2312";
  case 54936:
    return "GB18030";
  case 65000:
    return "UTF-7";
  case 65001:
    return "UTF-8";
  default:
    return nullptr;
  }
}
} // namespace

const char* __get_locale_encoding(__locale_t loc) {
  const char* locale_name = loc.__get_locale();
  if (locale_name == nullptr) {
    return __codepage_name(::GetACP());
  }

  std::string_view __sv(locale_name);

  // locale :: "locale-name"
  // | "language"[_country-region[.code-page]]
  // | ".code-page"
  // GetLocaleInfoEx doesn't accept anything other than BCP-47 locale names, e.g. "en_US",
  // so do a best-attempt to derive the text encoding from the name.
  if (__sv == "C" || __sv == "") {
    // "A locale argument value of C specifies the minimal ANSI conforming environment for C translation."
    // TODO: Figure out what to do for an empty string:
    // "If locale points to an empty string, the locale is the implementation-defined native environment."
    return __codepage_name(::GetACP());
  } else if (auto dot = __sv.find('.'); dot != std::string_view::npos) {
    std::string_view __code_page(locale_name + dot + 1);

    // Windows allows the codepage number as part of the name,
    // e.g. "en_US.1252" for English US, Windows-1252.
    if (std::isdigit(__code_page[0])) {
      unsigned int __cpage{};
      auto __res = std::from_chars(__code_page.data(), __code_page.data() + __code_page.size(), __cpage);
      if (__res) {
        return __codepage_name(__cpage);
      }
    } else { // POSIX-style name
      return locale_name + dot + 1;
    }
  }

  wchar_t locale_wbuffer[LOCALE_NAME_MAX_LENGTH + 1]{};
  wchar_t number_buffer[11]{};

  bool is_ansi  = ::AreFileApisANSI();
  auto codepage = is_ansi ? CP_ACP : CP_OEMCP;
  int ret       = ::MultiByteToWideChar(
      codepage, MB_ERR_INVALID_CHARS, locale_name, __sv.size(), locale_wbuffer, LOCALE_NAME_MAX_LENGTH);

  if (ret <= 0)
    return nullptr;

  // The below function fills the string with the number in text.
  auto lctype = is_ansi ? LOCALE_IDEFAULTANSICODEPAGE : LOCALE_IDEFAULTCODEPAGE;
  int result  = ::GetLocaleInfoEx(locale_wbuffer, lctype, number_buffer, 10);

  if (result <= 0)
    return nullptr;

  unsigned int acp = std::wcstoul(number_buffer, nullptr, 10);
  return __codepage_name(acp);
}

//
// Strtonum functions
//
#if !defined(_LIBCPP_MSVCRT)
float __strtof(const char* nptr, char** endptr, __locale_t loc) {
  __locale_guard __current(loc);
  return std::strtof(nptr, endptr);
}

long double __strtold(const char* nptr, char** endptr, __locale_t loc) {
  __locale_guard __current(loc);
  return std::strtold(nptr, endptr);
}
#endif

//
// Character manipulation functions
//
#if defined(__MINGW32__) && __MSVCRT_VERSION__ < 0x0800
size_t __strftime(char* ret, size_t n, const char* format, const struct tm* tm, __locale_t loc) {
  __locale_guard __current(loc);
  return std::strftime(ret, n, format, tm);
}
#endif

//
// Other functions
//
decltype(MB_CUR_MAX) __mb_len_max(__locale_t __l) {
#if defined(_LIBCPP_MSVCRT)
  return ::___mb_cur_max_l_func(__l);
#else
  __locale_guard __current(__l);
  return MB_CUR_MAX;
#endif
}

wint_t __btowc(int c, __locale_t loc) {
  __locale_guard __current(loc);
  return std::btowc(c);
}

int __wctob(wint_t c, __locale_t loc) {
  __locale_guard __current(loc);
  return std::wctob(c);
}

size_t __wcsnrtombs(char* __restrict dst,
                    const wchar_t** __restrict src,
                    size_t nwc,
                    size_t len,
                    mbstate_t* __restrict ps,
                    __locale_t loc) {
  __locale_guard __current(loc);
  return ::wcsnrtombs(dst, src, nwc, len, ps);
}

size_t __wcrtomb(char* __restrict s, wchar_t wc, mbstate_t* __restrict ps, __locale_t loc) {
  __locale_guard __current(loc);
  return std::wcrtomb(s, wc, ps);
}

size_t __mbsnrtowcs(wchar_t* __restrict dst,
                    const char** __restrict src,
                    size_t nms,
                    size_t len,
                    mbstate_t* __restrict ps,
                    __locale_t loc) {
  __locale_guard __current(loc);
  return ::mbsnrtowcs(dst, src, nms, len, ps);
}

size_t
__mbrtowc(wchar_t* __restrict pwc, const char* __restrict s, size_t n, mbstate_t* __restrict ps, __locale_t loc) {
  __locale_guard __current(loc);
  return std::mbrtowc(pwc, s, n, ps);
}

size_t __mbrlen(const char* __restrict s, size_t n, mbstate_t* __restrict ps, __locale_t loc) {
  __locale_guard __current(loc);
  return std::mbrlen(s, n, ps);
}

size_t __mbsrtowcs(
    wchar_t* __restrict dst, const char** __restrict src, size_t len, mbstate_t* __restrict ps, __locale_t loc) {
  __locale_guard __current(loc);
  return std::mbsrtowcs(dst, src, len, ps);
}

int __snprintf(char* ret, size_t n, __locale_t loc, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
#if defined(_LIBCPP_MSVCRT)
  // FIXME: Remove usage of internal CRT function and globals.
  int result = ::__stdio_common_vsprintf(
      _CRT_INTERNAL_LOCAL_PRINTF_OPTIONS | _CRT_INTERNAL_PRINTF_STANDARD_SNPRINTF_BEHAVIOR, ret, n, format, loc, ap);
#else
  __locale_guard __current(loc);
  _LIBCPP_DIAGNOSTIC_PUSH
  _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wformat-nonliteral")
  int result = std::vsnprintf(ret, n, format, ap);
  _LIBCPP_DIAGNOSTIC_POP
#endif
  va_end(ap);
  return result;
}

// Like sprintf, but when return value >= 0 it returns
// a pointer to a malloc'd string in *sptr.
// If return >= 0, use free to delete *sptr.
static int __libcpp_vasprintf(char** sptr, const char* __restrict format, va_list ap) {
  *sptr = nullptr;
  // Query the count required.
  va_list ap_copy;
  va_copy(ap_copy, ap);
  _LIBCPP_DIAGNOSTIC_PUSH
  _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wformat-nonliteral")
  int count = vsnprintf(nullptr, 0, format, ap_copy);
  _LIBCPP_DIAGNOSTIC_POP
  va_end(ap_copy);
  if (count < 0)
    return count;
  size_t buffer_size = static_cast<size_t>(count) + 1;
  char* p            = static_cast<char*>(malloc(buffer_size));
  if (!p)
    return -1;
  // If we haven't used exactly what was required, something is wrong.
  // Maybe bug in vsnprintf. Report the error and return.
  _LIBCPP_DIAGNOSTIC_PUSH
  _LIBCPP_CLANG_DIAGNOSTIC_IGNORED("-Wformat-nonliteral")
  if (vsnprintf(p, buffer_size, format, ap) != count) {
    _LIBCPP_DIAGNOSTIC_POP
    free(p);
    return -1;
  }
  // All good. This is returning memory to the caller not freeing it.
  *sptr = p;
  return count;
}

int __asprintf(char** ret, __locale_t loc, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  __locale_guard __current(loc);
  return __libcpp_vasprintf(ret, format, ap);
}

} // namespace __locale
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
