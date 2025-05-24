//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__config>
#include <__text_encoding/te_impl.h>

#if defined(_LIBCPP_WIN32API)
#  include <windows.h>
#else
#  include <__locale_dir/locale_base_api.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if defined(_LIBCPP_WIN32API)
_LIBCPP_HIDDEN __te_impl::__id __te_impl::__get_win32_acp() {
  switch (GetACP()) {
  case 037:
    return __te_impl::__id::IBM037;
  case 437:
    return __te_impl::__id::PC8CodePage437;
  case 500:
    return __te_impl::__id::IBM500;
  case 708:
    return __te_impl::__id::ISOLatinArabic;
  case 709:
    return __te_impl::__id::ISO89ASMO449;
  case 775:
    return __te_impl::__id::PC775Baltic;
  case 850:
    return __te_impl::__id::PC850Multilingual;
  case 852:
    return __te_impl::__id::PCp852;
  case 855:
    return __te_impl::__id::IBM855;
  case 857:
    return __te_impl::__id::IBM857;
  case 858:
    return __te_impl::__id::IBM00858;
  case 860:
    return __te_impl::__id::IBM860;
  case 861:
    return __te_impl::__id::IBM861;
  case 862:
    return __te_impl::__id::PC862LatinHebrew;
  case 863:
    return __te_impl::__id::IBM863;
  case 864:
    return __te_impl::__id::IBM864;
  case 865:
    return __te_impl::__id::IBM865;
  case 866:
    return __te_impl::__id::IBM866;
  case 869:
    return __te_impl::__id::IBM869;
  case 870:
    return __te_impl::__id::IBM870;
  case 874:
    return __te_impl::__id::windows874;
  case 932:
    return __te_impl::__id::ShiftJIS;
  case 936:
    return __te_impl::__id::GB2312;
  case 949:
    return __te_impl::__id::KSC56011987;
  case 950:
    return __te_impl::__id::Big5;
  case 1026:
    return __te_impl::__id::IBM1026;
  case 1047:
    return __te_impl::__id::IBM1047;
  case 1140:
    return __te_impl::__id::IBM01140;
  case 1141:
    return __te_impl::__id::IBM01141;
  case 1142:
    return __te_impl::__id::IBM01142;
  case 1143:
    return __te_impl::__id::IBM01143;
  case 1144:
    return __te_impl::__id::IBM01144;
  case 1145:
    return __te_impl::__id::IBM01145;
  case 1146:
    return __te_impl::__id::IBM01146;
  case 1147:
    return __te_impl::__id::IBM01147;
  case 1148:
    return __te_impl::__id::IBM01148;
  case 1149:
    return __te_impl::__id::IBM01149;
  case 1200:
    return __te_impl::__id::UTF16LE;
  case 1201:
    return __te_impl::__id::UTF16BE;
  case 1250:
    return __te_impl::__id::windows1250;
  case 1251:
    return __te_impl::__id::windows1251;
  case 1252:
    return __te_impl::__id::windows1252;
  case 1253:
    return __te_impl::__id::windows1253;
  case 1254:
    return __te_impl::__id::windows1254;
  case 1255:
    return __te_impl::__id::windows1255;
  case 1256:
    return __te_impl::__id::windows1256;
  case 1257:
    return __te_impl::__id::windows1257;
  case 1258:
    return __te_impl::__id::windows1258;
  case 10000:
    return __te_impl::__id::Macintosh;
  case 12000:
    return __te_impl::__id::UTF32LE;
  case 12001:
    return __te_impl::__id::UTF32BE;
  case 20127:
    return __te_impl::__id::ASCII;
  case 20273:
    return __te_impl::__id::IBM273;
  case 20277:
    return __te_impl::__id::IBM277;
  case 20278:
    return __te_impl::__id::IBM278;
  case 20280:
    return __te_impl::__id::IBM280;
  case 20284:
    return __te_impl::__id::IBM284;
  case 20285:
    return __te_impl::__id::IBM285;
  case 20290:
    return __te_impl::__id::IBM290;
  case 20297:
    return __te_impl::__id::IBM297;
  case 20420:
    return __te_impl::__id::IBM420;
  case 20423:
    return __te_impl::__id::IBM423;
  case 20424:
    return __te_impl::__id::IBM424;
  case 20838:
    return __te_impl::__id::IBMThai;
  case 20866:
    return __te_impl::__id::KOI8R;
  case 20871:
    return __te_impl::__id::IBM871;
  case 20880:
    return __te_impl::__id::IBM880;
  case 20905:
    return __te_impl::__id::IBM905;
  case 20924:
    return __te_impl::__id::IBM00924;
  case 20932:
    return __te_impl::__id::EUCPkdFmtJapanese;
  case 21866:
    return __te_impl::__id::KOI8U;
  case 28591:
    return __te_impl::__id::ISOLatin1;
  case 28592:
    return __te_impl::__id::ISOLatin2;
  case 28593:
    return __te_impl::__id::ISOLatin3;
  case 28594:
    return __te_impl::__id::ISOLatin4;
  case 28595:
    return __te_impl::__id::ISOLatin5;
  case 28596:
    return __te_impl::__id::ISOLatin6;
  case 28597:
    return __te_impl::__id::ISOLatinGreek;
  case 28598:
    return __te_impl::__id::ISOLatinHebrew;
  case 28599:
    return __te_impl::__id::Windows31Latin5;
  case 28603:
    return __te_impl::__id::ISO885913;
  case 28605:
    return __te_impl::__id::ISO885915;
  case 38598:
    return __te_impl::__id::ISO88598I;
  case 50220:
  case 50221:
  case 50222:
    return __te_impl::__id::ISO2022JP;
  case 51932:
    return __te_impl::__id::EUCPkdFmtJapanese;
  case 51936:
    return __te_impl::__id::GB2312;
  case 51949:
    return __te_impl::__id::EUCKR;
  case 52936:
    return __te_impl::__id::HZGB2312;
  case 54936:
    return __te_impl::__id::GB18030;
  case 65000:
    return __te_impl::__id::UTF7;
  case 65001:
    return __te_impl::__id::UTF8;
  default:
    return __te_impl::__id::other;
  }
}
#endif // _LIBCPP_WIN32API

#if !defined(__ANDROID__) && !defined(_LIBCPP_WIN32API)
_LIBCPP_HIDDEN __te_impl __te_impl::__get_locale_encoding(const char* __name) {
  __te_impl __e;

  __locale::__locale_t __l = __locale::__newlocale(_LIBCPP_CTYPE_MASK, __name, static_cast<__locale::__locale_t>(0));

  if (!__l) {
    return __e;
  }

  const char* __codeset = __locale::__nl_langinfo_l(_LIBCPP_NL_CODESET, __l);

  if (!__codeset) {
    return __e;
  }

  string_view __codeset_sv(__codeset);

  if (__codeset_sv.size() <= __te_impl::__max_name_length_) {
    __e = __te_impl(__codeset_sv);
  }

  __locale::__freelocale(__l);

  return __e;
}

#else
_LIBCPP_HIDDEN __te_impl __te_impl::__get_locale_encoding(const char* __name [[maybe_unused]]) { return __te_impl(); }
#endif

_LIBCPP_HIDDEN __te_impl __te_impl::__get_env_encoding() {
#if defined(_LIBCPP_WIN32API)
  return __te_impl(__get_win32_acp());
#else
  return __get_locale_encoding("");
#endif // _LIBCPP_WIN32API
}

_LIBCPP_AVAILABILITY_TE_ENVIRONMENT _LIBCPP_EXPORTED_FROM_ABI __te_impl __te_impl::__environment() {
  return __te_impl::__get_env_encoding();
}

_LIBCPP_END_NAMESPACE_STD
