//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__locale_dir/locale_base_api.h>
#include <text_encoding>
#if defined(_LIBCPP_WIN32API)
#  include <windows.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

__text_encoding_rep __text_encoding_rep::__get_locale_encoding(const char* __name) {
#if defined(_LIBCPP_WIN32API)
  auto _code_page = GetACP();
  auto __mib      = __text_encoding_rep::__id::other;

  switch (_code_page) {
    __mib = __text_encoding_rep::__id::IBM037;
    break;
  case 437:
    __mib = __text_encoding_rep::__id::PC8CodePage437;
    break;
  case 500:
    __mib = __text_encoding_rep::__id::IBM500;
    break;
  case 708:
    __mib = __text_encoding_rep::__id::ISOLatinArabic;
    break;
  case 709:
    __mib = __text_encoding_rep::__id::ISO89ASMO449;
    break;
  case 775:
    __mib = __text_encoding_rep::__id::PC775Baltic;
    break;
  case 850:
    __mib = __text_encoding_rep::__id::PC850Multilingual;
    break;
  case 852:
    __mib = __text_encoding_rep::__id::PCp852;
    break;
  case 855:
    __mib = __text_encoding_rep::__id::IBM855;
    break;
  case 857:
    __mib = __text_encoding_rep::__id::IBM857;
    break;
  case 858:
    __mib = __text_encoding_rep::__id::IBM00858;
    break;
  case 860:
    __mib = __text_encoding_rep::__id::IBM860;
    break;
  case 861:
    __mib = __text_encoding_rep::__id::IBM861;
    break;
  case 862:
    __mib = __text_encoding_rep::__id::PC862LatinHebrew;
    break;
  case 863:
    __mib = __text_encoding_rep::__id::IBM863;
    break;
  case 864:
    __mib = __text_encoding_rep::__id::IBM864;
    break;
  case 865:
    __mib = __text_encoding_rep::__id::IBM865;
    break;
  case 866:
    __mib = __text_encoding_rep::__id::IBM866;
    break;
  case 869:
    __mib = __text_encoding_rep::__id::IBM869;
    break;
  case 870:
    __mib = __text_encoding_rep::__id::IBM870;
    break;
  case 874:
    __mib = __text_encoding_rep::__id::windows874;
    break;
  case 932:
    __mib = __text_encoding_rep::__id::ShiftJIS;
    break;
  case 936:
    __mib = __text_encoding_rep::__id::GB2312;
    break;
  case 949:
    __mib = __text_encoding_rep::__id::KSC56011987;
    break;
  case 950:
    __mib = __text_encoding_rep::__id::Big5;
    break;
  case 1026:
    __mib = __text_encoding_rep::__id::IBM1026;
    break;
  case 1047:
    __mib = __text_encoding_rep::__id::IBM1047;
    break;
  case 1140:
    __mib = __text_encoding_rep::__id::IBM01140;
    break;
  case 1141:
    __mib = __text_encoding_rep::__id::IBM01141;
    break;
  case 1142:
    __mib = __text_encoding_rep::__id::IBM01142;
    break;
  case 1143:
    __mib = __text_encoding_rep::__id::IBM01143;
    break;
  case 1144:
    __mib = __text_encoding_rep::__id::IBM01144;
    break;
  case 1145:
    __mib = __text_encoding_rep::__id::IBM01145;
    break;
  case 1146:
    __mib = __text_encoding_rep::__id::IBM01146;
    break;
  case 1147:
    __mib = __text_encoding_rep::__id::IBM01147;
    break;
  case 1148:
    __mib = __text_encoding_rep::__id::IBM01148;
    break;
  case 1149:
    __mib = __text_encoding_rep::__id::IBM01149;
    break;
  case 1200:
    __mib = __text_encoding_rep::__id::UTF16LE;
    break;
  case 1201:
    __mib = __text_encoding_rep::__id::UTF16BE;
    break;
  case 1250:
    __mib = __text_encoding_rep::__id::windows1250;
    break;
  case 1251:
    __mib = __text_encoding_rep::__id::windows1251;
    break;
  case 1252:
    __mib = __text_encoding_rep::__id::windows1252;
    break;
  case 1253:
    __mib = __text_encoding_rep::__id::windows1253;
    break;
  case 1254:
    __mib = __text_encoding_rep::__id::windows1254;
    break;
  case 1255:
    __mib = __text_encoding_rep::__id::windows1255;
    break;
  case 1256:
    __mib = __text_encoding_rep::__id::windows1256;
    break;
  case 1257:
    __mib = __text_encoding_rep::__id::windows1257;
    break;
  case 1258:
    __mib = __text_encoding_rep::__id::windows1258;
    break;
  case 10000:
    __mib = __text_encoding_rep::__id::Macintosh;
    break;
  case 12000:
    __mib = __text_encoding_rep::__id::UTF32LE;
    break;
  case 12001:
    __mib = __text_encoding_rep::__id::UTF32BE;
    break;
  case 20127:
    __mib = __text_encoding_rep::__id::ASCII;
    break;
  case 20273:
    __mib = __text_encoding_rep::__id::IBM273;
    break;
  case 20277:
    __mib = __text_encoding_rep::__id::IBM277;
    break;
  case 20278:
    __mib = __text_encoding_rep::__id::IBM278;
    break;
  case 20280:
    __mib = __text_encoding_rep::__id::IBM280;
    break;
  case 20284:
    __mib = __text_encoding_rep::__id::IBM284;
    break;
  case 20285:
    __mib = __text_encoding_rep::__id::IBM285;
    break;
  case 20290:
    __mib = __text_encoding_rep::__id::IBM290;
    break;
  case 20297:
    __mib = __text_encoding_rep::__id::IBM297;
    break;
  case 20420:
    __mib = __text_encoding_rep::__id::IBM420;
    break;
  case 20423:
    __mib = __text_encoding_rep::__id::IBM423;
    break;
  case 20424:
    __mib = __text_encoding_rep::__id::IBM424;
    break;
  case 20838:
    __mib = __text_encoding_rep::__id::IBMThai;
    break;
  case 20866:
    __mib = __text_encoding_rep::__id::KOI8R;
    break;
  case 20871:
    __mib = __text_encoding_rep::__id::IBM871;
    break;
  case 20880:
    __mib = __text_encoding_rep::__id::IBM880;
    break;
  case 20905:
    __mib = __text_encoding_rep::__id::IBM905;
    break;
  case 20924:
    __mib = __text_encoding_rep::__id::IBM00924;
    break;
  case 20932:
    __mib = __text_encoding_rep::__id::EUCPkdFmtJapanese;
    break;
  case 21866:
    __mib = __text_encoding_rep::__id::KOI8U;
    break;
  case 28591:
    __mib = __text_encoding_rep::__id::ISOLatin1;
    break;
  case 28592:
    __mib = __text_encoding_rep::__id::ISOLatin2;
    break;
  case 28593:
    __mib = __text_encoding_rep::__id::ISOLatin3;
    break;
  case 28594:
    __mib = __text_encoding_rep::__id::ISOLatin4;
    break;
  case 28595:
    __mib = __text_encoding_rep::__id::ISOLatin5;
    break;
  case 28596:
    __mib = __text_encoding_rep::__id::ISOLatin6;
    break;
  case 28597:
    __mib = __text_encoding_rep::__id::ISOLatinGreek;
    break;
  case 28598:
    __mib = __text_encoding_rep::__id::ISOLatinHebrew;
    break;
  case 28599:
    __mib = __text_encoding_rep::__id::Windows31Latin5;
    break;
  case 28603:
    __mib = __text_encoding_rep::__id::ISO885913;
    break;
  case 28605:
    __mib = __text_encoding_rep::__id::ISO885915;
    break;
  case 38598:
    __mib = __text_encoding_rep::__id::ISO88598I;
    break;
  case 50220:
    __mib = __text_encoding_rep::__id::ISO2022JP;
    break;
  case 50221:
    __mib = __text_encoding_rep::__id::ISO2022JP;
    break;
  case 50222:
    __mib = __text_encoding_rep::__id::ISO2022JP;
    break;
  case 51932:
    __mib = __text_encoding_rep::__id::EUCPkdFmtJapanese;
    break;
  case 51936:
    __mib = __text_encoding_rep::__id::GB2312;
    break;
  case 51949:
    __mib = __text_encoding_rep::__id::EUCKR;
    break;
  case 52936:
    __mib = __text_encoding_rep::__id::HZGB2312;
    break;
  case 54936:
    __mib = __text_encoding_rep::__id::GB18030;
    break;
  case 65000:
    __mib = __text_encoding_rep::__id::UTF7;
    break;
  case 65001:
    __mib = __text_encoding_rep::__id::UTF8;
    break;
  default:
    __mib = __text_encoding_rep::__id::other;
    break;
  };
  return __text_encoding_rep(__mib);
#else
  __text_encoding_rep __encoding{};
  if (auto __loc = __locale::__newlocale(_LIBCPP_CTYPE_MASK, __name, static_cast<__locale::__locale_t>(0))) {
    if (const char* __codeset = __locale::__nl_langinfo_l(CODESET, __loc)) {
      string_view __s(__codeset);
      if (__s.size() <= __text_encoding_rep::__max_name_length_)
        __encoding = __text_encoding_rep(__s);
    }
    __locale::__freelocale(__loc);
  }
  return __encoding;
#endif
}

_LIBCPP_END_NAMESPACE_STD
