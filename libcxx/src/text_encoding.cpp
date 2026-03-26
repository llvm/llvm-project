//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <text_encoding>

#if defined(_LIBCPP_WIN32API)
#  include <__algorithm/max.h>
#  include <cwchar>
#  include <windows.h>
#else
#  include <__locale_dir/locale_base_api.h>
#  include <__utility/scope_guard.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if defined(_LIBCPP_WIN32API)
_LIBCPP_HIDDEN text_encoding __get_win32_acp(unsigned int __codepage);

_LIBCPP_HIDDEN text_encoding static __get_win32_acp(unsigned int __codepage) {
  switch (__codepage) {
  case 0:
    // If no ANSI code page is available, only Unicode can be used for the locale.
    // In this case, the value is CP_ACP (0).
    // Such a locale cannot be set as the system locale.
    // Applications that do not support Unicode do not work correctly with locales
    // marked as "Unicode only".
    return std::text_encoding::id::unknown;
  case 037:
    return std::text_encoding::id::IBM037;
  case 437:
    return std::text_encoding::id::PC8CodePage437;
  case 500:
    return std::text_encoding::id::IBM500;
  case 708:
    return std::text_encoding::id::ISOLatinArabic;
  case 709:
    return std::text_encoding::id::ISO89ASMO449;
  case 775:
    return std::text_encoding::id::PC775Baltic;
  case 850:
    return std::text_encoding::id::PC850Multilingual;
  case 852:
    return std::text_encoding::id::PCp852;
  case 855:
    return std::text_encoding::id::IBM855;
  case 857:
    return std::text_encoding::id::IBM857;
  case 858:
    return std::text_encoding::id::IBM00858;
  case 860:
    return std::text_encoding::id::IBM860;
  case 861:
    return std::text_encoding::id::IBM861;
  case 862:
    return std::text_encoding::id::PC862LatinHebrew;
  case 863:
    return std::text_encoding::id::IBM863;
  case 864:
    return std::text_encoding::id::IBM864;
  case 865:
    return std::text_encoding::id::IBM865;
  case 866:
    return std::text_encoding::id::IBM866;
  case 869:
    return std::text_encoding::id::IBM869;
  case 870:
    return std::text_encoding::id::IBM870;
  case 874:
    return std::text_encoding::id::windows874;
  case 932:
    return std::text_encoding::id::ShiftJIS;
  case 936:
    return std::text_encoding::id::GB2312;
  case 949:
    return std::text_encoding::id::KSC56011987;
  case 950:
    return std::text_encoding::id::Big5;
  case 1026:
    return std::text_encoding::id::IBM1026;
  case 1047:
    return std::text_encoding::id::IBM1047;
  case 1140:
    return std::text_encoding::id::IBM01140;
  case 1141:
    return std::text_encoding::id::IBM01141;
  case 1142:
    return std::text_encoding::id::IBM01142;
  case 1143:
    return std::text_encoding::id::IBM01143;
  case 1144:
    return std::text_encoding::id::IBM01144;
  case 1145:
    return std::text_encoding::id::IBM01145;
  case 1146:
    return std::text_encoding::id::IBM01146;
  case 1147:
    return std::text_encoding::id::IBM01147;
  case 1148:
    return std::text_encoding::id::IBM01148;
  case 1149:
    return std::text_encoding::id::IBM01149;
  case 1200:
    return std::text_encoding::id::UTF16LE;
  case 1201:
    return std::text_encoding::id::UTF16BE;
  case 1250:
    return std::text_encoding::id::windows1250;
  case 1251:
    return std::text_encoding::id::windows1251;
  case 1252:
    return std::text_encoding::id::windows1252;
  case 1253:
    return std::text_encoding::id::windows1253;
  case 1254:
    return std::text_encoding::id::windows1254;
  case 1255:
    return std::text_encoding::id::windows1255;
  case 1256:
    return std::text_encoding::id::windows1256;
  case 1257:
    return std::text_encoding::id::windows1257;
  case 1258:
    return std::text_encoding::id::windows1258;
  case 10000:
    return std::text_encoding::id::Macintosh;
  case 12000:
    return std::text_encoding::id::UTF32LE;
  case 12001:
    return std::text_encoding::id::UTF32BE;
  case 20127:
    return std::text_encoding::id::ASCII;
  case 20273:
    return std::text_encoding::id::IBM273;
  case 20277:
    return std::text_encoding::id::IBM277;
  case 20278:
    return std::text_encoding::id::IBM278;
  case 20280:
    return std::text_encoding::id::IBM280;
  case 20284:
    return std::text_encoding::id::IBM284;
  case 20285:
    return std::text_encoding::id::IBM285;
  case 20290:
    return std::text_encoding::id::IBM290;
  case 20297:
    return std::text_encoding::id::IBM297;
  case 20420:
    return std::text_encoding::id::IBM420;
  case 20423:
    return std::text_encoding::id::IBM423;
  case 20424:
    return std::text_encoding::id::IBM424;
  case 20838:
    return std::text_encoding::id::IBMThai;
  case 20866:
    return std::text_encoding::id::KOI8R;
  case 20871:
    return std::text_encoding::id::IBM871;
  case 20880:
    return std::text_encoding::id::IBM880;
  case 20905:
    return std::text_encoding::id::IBM905;
  case 20924:
    return std::text_encoding::id::IBM00924;
  case 20932:
    return std::text_encoding::id::EUCPkdFmtJapanese;
  case 21866:
    return std::text_encoding::id::KOI8U;
  case 28591:
    return std::text_encoding::id::ISOLatin1;
  case 28592:
    return std::text_encoding::id::ISOLatin2;
  case 28593:
    return std::text_encoding::id::ISOLatin3;
  case 28594:
    return std::text_encoding::id::ISOLatin4;
  case 28595:
    return std::text_encoding::id::ISOLatin5;
  case 28596:
    return std::text_encoding::id::ISOLatin6;
  case 28597:
    return std::text_encoding::id::ISOLatinGreek;
  case 28598:
    return std::text_encoding::id::ISOLatinHebrew;
  case 28599:
    return std::text_encoding::id::Windows31Latin5;
  case 28603:
    return std::text_encoding::id::ISO885913;
  case 28605:
    return std::text_encoding::id::ISO885915;
  case 38598:
    return std::text_encoding::id::ISO88598I;
  case 50220:
  case 50221:
  case 50222:
    return std::text_encoding::id::ISO2022JP;
  case 51932:
    return std::text_encoding::id::EUCPkdFmtJapanese;
  case 51936:
    return std::text_encoding::id::GB2312;
  case 51949:
    return std::text_encoding::id::EUCKR;
  case 52936:
    return std::text_encoding::id::HZGB2312;
  case 54936:
    return std::text_encoding::id::GB18030;
  case 65000:
    return std::text_encoding::id::UTF7;
  case 65001:
    return std::text_encoding::id::UTF8;
  default:
    return std::text_encoding::id::unknown;
  }
}

_LIBCPP_EXPORTED_FROM_ABI std::text_encoding __get_locale_encoding(const char* __name) {
  wchar_t __locale_wbuffer[LOCALE_NAME_MAX_LENGTH + 1]{};
  wchar_t __number_buffer[11]{};

  bool __is_ansi  = ::AreFileApisANSI();
  auto __codepage = __is_ansi ? CP_ACP : CP_OEMCP;

  string_view __sv(__name);
  int __ret = ::MultiByteToWideChar(
      __codepage, MB_ERR_INVALID_CHARS, __name, __sv.size(), __locale_wbuffer, LOCALE_NAME_MAX_LENGTH);

  if (__ret <= 0)
    return std::text_encoding();

  // The below function fills the string with the number in text.
  auto __lctype = __is_ansi ? LOCALE_IDEFAULTANSICODEPAGE : LOCALE_IDEFAULTCODEPAGE;
  int __result  = ::GetLocaleInfoEx(__locale_wbuffer, __lctype, __number_buffer, 10);

  if (__result <= 0)
    return std::text_encoding();

  unsigned int __acp = std::wcstoul(__number_buffer, nullptr, 10);

  return __get_win32_acp(__acp);
}

_LIBCPP_HIDDEN static std::text_encoding __get_env_encoding() { return __get_win32_acp(::GetACP()); }

#elif defined(__ANDROID__)
// Android has minimal libc suppport for locale, and doesn't support any other locale
// than the ones checked for below.
_LIBCPP_EXPORTED_FROM_ABI std::text_encoding __get_locale_encoding(const char* __name) {
  string_view __sv(__name);
  if (__sv == "" || __sv == "*" || __sv == "C" || __sv == "POSIX" || __sv.contains("UTF-8")) {
    return std::text_encoding(std::text_encoding::id::UTF8);
  }

  return std::text_encoding();
}

// Android is pretty much assumed to always be UTF-8.
_LIBCPP_HIDDEN static std::text_encoding __get_env_encoding() {
  return std::text_encoding(std::text_encoding::id::UTF8);
}

#else  // POSIX
_LIBCPP_EXPORTED_FROM_ABI std::text_encoding __get_locale_encoding(const char* __name) {
  std::text_encoding __e;

  __locale::__locale_t __l = __locale::__newlocale(_LIBCPP_CTYPE_MASK, __name, static_cast<__locale::__locale_t>(0));

  __scope_guard __locale_guard([&__l] {
    if (__l) {
      __locale::__freelocale(__l);
    }
  });

  if (!__l) {
    return __e;
  }

  const char* __codeset = __locale::__nl_langinfo(_LIBCPP_NL_CODESET, __l);

  if (!__codeset) {
    return __e;
  }

  string_view __codeset_sv(__codeset);

  if (__codeset_sv.size() <= std::text_encoding::max_name_length) {
    __e = std::text_encoding(__codeset_sv);
  }

  return __e;
}

_LIBCPP_HIDDEN static std::text_encoding __get_env_encoding() { return __get_locale_encoding(""); }
#endif // _LIBCPP_WIN32API

_LIBCPP_AVAILABILITY_TE_ENVIRONMENT _LIBCPP_EXPORTED_FROM_ABI std::text_encoding std::text_encoding::environment() {
  return __get_env_encoding();
}

_LIBCPP_END_NAMESPACE_STD
