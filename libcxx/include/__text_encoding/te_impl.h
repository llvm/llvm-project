// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP__TEXT_ENCODING_TE_IMPL_H
#define _LIBCPP__TEXT_ENCODING_TE_IMPL_H

#include <__algorithm/copy_n.h>
#include <__algorithm/find.h>
#include <__algorithm/lower_bound.h>
#include <__config>
#include <__cstddef/ptrdiff_t.h>
#include <__iterator/iterator_traits.h>
#include <__ranges/view_interface.h>
#include <cstdint>
#include <string_view>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

struct __te_impl {
private:
  friend struct text_encoding;
  enum class __id : int_least32_t {
    other                   = 1,
    unknown                 = 2,
    ASCII                   = 3,
    ISOLatin1               = 4,
    ISOLatin2               = 5,
    ISOLatin3               = 6,
    ISOLatin4               = 7,
    ISOLatinCyrillic        = 8,
    ISOLatinArabic          = 9,
    ISOLatinGreek           = 10,
    ISOLatinHebrew          = 11,
    ISOLatin5               = 12,
    ISOLatin6               = 13,
    ISOTextComm             = 14,
    HalfWidthKatakana       = 15,
    JISEncoding             = 16,
    ShiftJIS                = 17,
    EUCPkdFmtJapanese       = 18,
    EUCFixWidJapanese       = 19,
    ISO4UnitedKingdom       = 20,
    ISO11SwedishForNames    = 21,
    ISO15Italian            = 22,
    ISO17Spanish            = 23,
    ISO21German             = 24,
    ISO60DanishNorwegian    = 25,
    ISO69French             = 26,
    ISO10646UTF1            = 27,
    ISO646basic1983         = 28,
    INVARIANT               = 29,
    ISO2IntlRefVersion      = 30,
    NATSSEFI                = 31,
    NATSSEFIADD             = 32,
    ISO10Swedish            = 35,
    KSC56011987             = 36,
    ISO2022KR               = 37,
    EUCKR                   = 38,
    ISO2022JP               = 39,
    ISO2022JP2              = 40,
    ISO13JISC6220jp         = 41,
    ISO14JISC6220ro         = 42,
    ISO16Portuguese         = 43,
    ISO18Greek7Old          = 44,
    ISO19LatinGreek         = 45,
    ISO25French             = 46,
    ISO27LatinGreek1        = 47,
    ISO5427Cyrillic         = 48,
    ISO42JISC62261978       = 49,
    ISO47BSViewdata         = 50,
    ISO49INIS               = 51,
    ISO50INIS8              = 52,
    ISO51INISCyrillic       = 53,
    ISO54271981             = 54,
    ISO5428Greek            = 55,
    ISO57GB1988             = 56,
    ISO58GB231280           = 57,
    ISO61Norwegian2         = 58,
    ISO70VideotexSupp1      = 59,
    ISO84Portuguese2        = 60,
    ISO85Spanish2           = 61,
    ISO86Hungarian          = 62,
    ISO87JISX0208           = 63,
    ISO88Greek7             = 64,
    ISO89ASMO449            = 65,
    ISO90                   = 66,
    ISO91JISC62291984a      = 67,
    ISO92JISC62991984b      = 68,
    ISO93JIS62291984badd    = 69,
    ISO94JIS62291984hand    = 70,
    ISO95JIS62291984handadd = 71,
    ISO96JISC62291984kana   = 72,
    ISO2033                 = 73,
    ISO99NAPLPS             = 74,
    ISO102T617bit           = 75,
    ISO103T618bit           = 76,
    ISO111ECMACyrillic      = 77,
    ISO121Canadian1         = 78,
    ISO122Canadian2         = 79,
    ISO123CSAZ24341985gr    = 80,
    ISO88596E               = 81,
    ISO88596I               = 82,
    ISO128T101G2            = 83,
    ISO88598E               = 84,
    ISO88598I               = 85,
    ISO139CSN369103         = 86,
    ISO141JUSIB1002         = 87,
    ISO143IECP271           = 88,
    ISO146Serbian           = 89,
    ISO147Macedonian        = 90,
    ISO150                  = 91,
    ISO151Cuba              = 92,
    ISO6937Add              = 93,
    ISO153GOST1976874       = 94,
    ISO8859Supp             = 95,
    ISO10367Box             = 96,
    ISO158Lap               = 97,
    ISO159JISX02121990      = 98,
    ISO646Danish            = 99,
    USDK                    = 100,
    DKUS                    = 101,
    KSC5636                 = 102,
    Unicode11UTF7           = 103,
    ISO2022CN               = 104,
    ISO2022CNEXT            = 105,
    UTF8                    = 106,
    ISO885913               = 109,
    ISO885914               = 110,
    ISO885915               = 111,
    ISO885916               = 112,
    GBK                     = 113,
    GB18030                 = 114,
    OSDEBCDICDF0415         = 115,
    OSDEBCDICDF03IRV        = 116,
    OSDEBCDICDF041          = 117,
    ISO115481               = 118,
    KZ1048                  = 119,
    UCS2                    = 1000,
    UCS4                    = 1001,
    UnicodeASCII            = 1002,
    UnicodeLatin1           = 1003,
    UnicodeJapanese         = 1004,
    UnicodeIBM1261          = 1005,
    UnicodeIBM1268          = 1006,
    UnicodeIBM1276          = 1007,
    UnicodeIBM1264          = 1008,
    UnicodeIBM1265          = 1009,
    Unicode11               = 1010,
    SCSU                    = 1011,
    UTF7                    = 1012,
    UTF16BE                 = 1013,
    UTF16LE                 = 1014,
    UTF16                   = 1015,
    CESU8                   = 1016,
    UTF32                   = 1017,
    UTF32BE                 = 1018,
    UTF32LE                 = 1019,
    BOCU1                   = 1020,
    UTF7IMAP                = 1021,
    Windows30Latin1         = 2000,
    Windows31Latin1         = 2001,
    Windows31Latin2         = 2002,
    Windows31Latin5         = 2003,
    HPRoman8                = 2004,
    AdobeStandardEncoding   = 2005,
    VenturaUS               = 2006,
    VenturaInternational    = 2007,
    DECMCS                  = 2008,
    PC850Multilingual       = 2009,
    PC8DanishNorwegian      = 2012,
    PC862LatinHebrew        = 2013,
    PC8Turkish              = 2014,
    IBMSymbols              = 2015,
    IBMThai                 = 2016,
    HPLegal                 = 2017,
    HPPiFont                = 2018,
    HPMath8                 = 2019,
    HPPSMath                = 2020,
    HPDesktop               = 2021,
    VenturaMath             = 2022,
    MicrosoftPublishing     = 2023,
    Windows31J              = 2024,
    GB2312                  = 2025,
    Big5                    = 2026,
    Macintosh               = 2027,
    IBM037                  = 2028,
    IBM038                  = 2029,
    IBM273                  = 2030,
    IBM274                  = 2031,
    IBM275                  = 2032,
    IBM277                  = 2033,
    IBM278                  = 2034,
    IBM280                  = 2035,
    IBM281                  = 2036,
    IBM284                  = 2037,
    IBM285                  = 2038,
    IBM290                  = 2039,
    IBM297                  = 2040,
    IBM420                  = 2041,
    IBM423                  = 2042,
    IBM424                  = 2043,
    PC8CodePage437          = 2011,
    IBM500                  = 2044,
    IBM851                  = 2045,
    PCp852                  = 2010,
    IBM855                  = 2046,
    IBM857                  = 2047,
    IBM860                  = 2048,
    IBM861                  = 2049,
    IBM863                  = 2050,
    IBM864                  = 2051,
    IBM865                  = 2052,
    IBM868                  = 2053,
    IBM869                  = 2054,
    IBM870                  = 2055,
    IBM871                  = 2056,
    IBM880                  = 2057,
    IBM891                  = 2058,
    IBM903                  = 2059,
    IBM904                  = 2060,
    IBM905                  = 2061,
    IBM918                  = 2062,
    IBM1026                 = 2063,
    IBMEBCDICATDE           = 2064,
    EBCDICATDEA             = 2065,
    EBCDICCAFR              = 2066,
    EBCDICDKNO              = 2067,
    EBCDICDKNOA             = 2068,
    EBCDICFISE              = 2069,
    EBCDICFISEA             = 2070,
    EBCDICFR                = 2071,
    EBCDICIT                = 2072,
    EBCDICPT                = 2073,
    EBCDICES                = 2074,
    EBCDICESA               = 2075,
    EBCDICESS               = 2076,
    EBCDICUK                = 2077,
    EBCDICUS                = 2078,
    Unknown8BiT             = 2079,
    Mnemonic                = 2080,
    Mnem                    = 2081,
    VISCII                  = 2082,
    VIQR                    = 2083,
    KOI8R                   = 2084,
    HZGB2312                = 2085,
    IBM866                  = 2086,
    PC775Baltic             = 2087,
    KOI8U                   = 2088,
    IBM00858                = 2089,
    IBM00924                = 2090,
    IBM01140                = 2091,
    IBM01141                = 2092,
    IBM01142                = 2093,
    IBM01143                = 2094,
    IBM01144                = 2095,
    IBM01145                = 2096,
    IBM01146                = 2097,
    IBM01147                = 2098,
    IBM01148                = 2099,
    IBM01149                = 2100,
    Big5HKSCS               = 2101,
    IBM1047                 = 2102,
    PTCP154                 = 2103,
    Amiga1251               = 2104,
    KOI7switched            = 2105,
    BRF                     = 2106,
    TSCII                   = 2107,
    CP51932                 = 2108,
    windows874              = 2109,
    windows1250             = 2250,
    windows1251             = 2251,
    windows1252             = 2252,
    windows1253             = 2253,
    windows1254             = 2254,
    windows1255             = 2255,
    windows1256             = 2256,
    windows1257             = 2257,
    windows1258             = 2258,
    TIS620                  = 2259,
    CP50220                 = 2260
  };

  using enum __id;
  static constexpr size_t __max_name_length_ = 63;

  struct __te_data {
    const char* __name_;
    int_least32_t __mib_rep_;
    uint16_t __name_size_;
    uint16_t __to_end_;
    // The encoding data knows the end of its range to simplify iterator implementation.

    _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __te_data& __e, const __te_data& __other) noexcept {
      return __e.__mib_rep_ == __other.__mib_rep_ ||
             __comp_name(string_view(__e.__name_, __e.__name_size_), string_view(__other.__name_, __e.__name_size_));
    }

    _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __te_data& __e, const int_least32_t __i) noexcept {
      return __e.__mib_rep_ < __i;
    }

    _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __te_data& __e, std::string_view __name) noexcept {
      return __comp_name(__name, string_view(__e.__name_, __e.__name_size_));
    }
  }; // __te_data

  _LIBCPP_HIDE_FROM_ABI static constexpr bool __comp_name(string_view __a, string_view __b) {
    if (__a.empty() || __b.empty()) {
      return false;
    }

    // Map any non-alphanumeric character to 255, skip prefix 0s, else get tolower(__n).
    auto __map_char = [](char __n, bool& __in_number) -> unsigned char {
      auto __to_lower = [](char __n_in) -> char {
        return (__n_in >= 'A' && __n_in <= 'Z') ? __n_in + ('a' - 'A') : __n_in;
      };
      if (__n == '0') {
        return __in_number ? '0' : 255;
      }
      __in_number = __n >= '1' && __n <= '9';
      return (__n >= '1' && __n <= '9') || (__n >= 'A' && __n <= 'Z') || (__n >= 'a' && __n <= 'z')
               ? __to_lower(__n)
               : 255;
    };

    auto __a_ptr = __a.begin(), __b_ptr = __b.begin();
    bool __a_in_number = false, __b_in_number = false;

    unsigned char __a_val = 255, __b_val = 255;
    for (;; __a_ptr++, __b_ptr++) {
      while (__a_ptr != __a.end() && (__a_val = __map_char(*__a_ptr, __a_in_number)) == 255)
        __a_ptr++;
      while (__b_ptr != __b.end() && (__b_val = __map_char(*__b_ptr, __b_in_number)) == 255)
        __b_ptr++;

      if (__a_ptr == __a.end())
        return __b_ptr == __b.end();
      if (__b_ptr == __b.end())
        return false;
      if (__a_val != __b_val)
        return false;
    }
    return true;
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr const __te_data* __find_encoding_data(string_view __a) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        __a.size() <= __max_name_length_ && !__a.contains('\0'), "invalid string passed to text_encoding(string_view)");
    const __te_data* __data_first = __text_encoding_data + 2;
    const __te_data* __data_last  = std::end(__text_encoding_data);

    auto* __found_data = std::find(__data_first, __data_last, __a);

    if (__found_data == __data_last) {
      return __text_encoding_data; // other
    }

    while (__found_data[-1].__mib_rep_ == __found_data->__mib_rep_) {
      __found_data--;
    }

    return __found_data;
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr const __te_data* __find_encoding_data_by_id(__id __i) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(
        ((__i >= __id::other && __i <= __id::CP50220) && ((int_least32_t(__i) != 33) && (int_least32_t(__i) != 34))),
        "invalid text_encoding::id passed to text_encoding(id)");
    auto __found =
        std::lower_bound(std::begin(__text_encoding_data), std::end(__text_encoding_data), int_least32_t(__i));

    return __found != std::end(__text_encoding_data)
             ? __found
             : __text_encoding_data + 1; // unknown, should be unreachable
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __te_impl() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __te_impl(string_view __enc) noexcept
      : __encoding_rep_(__find_encoding_data(__enc)) {
    __enc.copy(__name_, __max_name_length_, 0);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __te_impl(__id __i) noexcept : __encoding_rep_(__find_encoding_data_by_id(__i)) {
    if (__encoding_rep_->__name_[0] != '\0')
      std::copy_n(__encoding_rep_->__name_, __encoding_rep_->__name_size_, __name_);
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr __id __mib() const noexcept {
    return __id(__encoding_rep_->__mib_rep_);
  }
  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr const char* __name() const noexcept { return __name_; }

  // [text.encoding.aliases], class text_encoding::aliases_view
  struct __aliases_view : ranges::view_interface<__aliases_view> {
    struct __iterator {
      using iterator_concept  = random_access_iterator_tag;
      using iterator_category = random_access_iterator_tag;
      using value_type        = const char*;
      using reference         = const char*;
      using difference_type   = ptrdiff_t;

      constexpr __iterator() noexcept = default;

      _LIBCPP_HIDE_FROM_ABI constexpr value_type operator*() const { return __data_->__name_; }

      _LIBCPP_HIDE_FROM_ABI constexpr value_type operator[](difference_type __n) const {
        auto __it = *this;
        return *(__it + __n);
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(__iterator __it, difference_type __n) {
        __it += __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, __iterator __it) {
        __it += __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(__iterator __it, difference_type __n) {
        __it -= __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr difference_type operator-(const __iterator& __other) const {
        return __data_ - __other.__data_;
      }

      _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(difference_type __n, __iterator& __it) {
        __it -= __n;
        return __it;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
        __data_++;
        return *this;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int) {
        auto __old = *this;
        __data_++;
        return __old;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--() {
        __data_--;
        return *this;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int) {
        auto __old = *this;
        __data_--;
        return __old;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n) {
        __data_ += __n;
        return *this;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n) { return operator+=(-__n); }

      _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const __iterator& __it) const { return __data_ == __it.__data_; }

      _LIBCPP_HIDE_FROM_ABI constexpr auto operator<=>(__iterator __it) const { return __data_ <=> __it.__data_; }

    private:
      friend struct __aliases_view;

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator(const __te_data* __enc_d) noexcept : __data_(__enc_d) {}

      const __te_data* __data_;
    }; // __iterator

    _LIBCPP_HIDE_FROM_ABI constexpr __iterator begin() const { return __iterator{__view_data_}; }
    _LIBCPP_HIDE_FROM_ABI constexpr __iterator end() const {
      return __iterator{__view_data_ + __view_data_->__to_end_};
    }

  private:
    friend struct __te_impl;

    _LIBCPP_HIDE_FROM_ABI constexpr __aliases_view(const __te_data* __d) : __view_data_(__d) {}
    const __te_data* __view_data_;
  };

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr __aliases_view __aliases() const {
    return __aliases_view(__encoding_rep_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __te_impl& __a, const __te_impl& __b) noexcept {
    return __a.__mib() == __id::other && __b.__mib() == __id::other
             ? __comp_name(__a.__name_, __b.__name_)
             : __a.__mib() == __b.__mib();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __te_impl& __encoding, const __id __i) noexcept {
    return __encoding.__mib() == __i;
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI static consteval __te_impl __literal() noexcept {
    // TODO: Remove this branch once we have __GNUC_EXECUTION_CHARSET_NAME or __clang_literal_encoding__ unconditionally
#  ifdef __GNUC_EXECUTION_CHARSET_NAME
    return __te_impl(__GNUC_EXECUTION_CHARSET_NAME);
#  elif defined(__clang_literal_encoding__)
    return __te_impl(__clang_literal_encoding__);
#  else
    return __te_impl();
#  endif
  }

#  if _LIBCPP_HAS_LOCALIZATION
  _LIBCPP_HIDDEN static __te_impl __get_locale_encoding(const char* __name);
  _LIBCPP_HIDDEN static __te_impl __get_env_encoding();
#    if defined(_LIBCPP_WIN32API)
  _LIBCPP_HIDDEN static __id __get_win32_acp();
#    endif

  [[__nodiscard__]] _LIBCPP_AVAILABILITY_TE_ENVIRONMENT _LIBCPP_EXPORTED_FROM_ABI static __te_impl __environment();

  template <__id _Id>
  [[__nodiscard__]] _LIBCPP_AVAILABILITY_TE_ENVIRONMENT _LIBCPP_HIDE_FROM_ABI static bool __environment_is() {
    return __environment() == _Id;
  }

#  endif

  const __te_data* __encoding_rep_     = __text_encoding_data + 1;
  char __name_[__max_name_length_ + 1] = {0};

  _LIBCPP_HIDE_FROM_ABI static constexpr __te_data __text_encoding_data[] = {
      {"", 1, 0, 0}, // other
      {"", 2, 0, 0}, // unknown
      {"US-ASCII", 3, 8, 11},
      {"iso-ir-6", 3, 8, 10},
      {"ANSI_X3.4-1968", 3, 14, 9},
      {"ANSI_X3.4-1986", 3, 14, 8},
      {"ISO_646.irv:1991", 3, 16, 7},
      {"ISO646-US", 3, 9, 6},
      {"us", 3, 2, 5},
      {"IBM367", 3, 6, 4},
      {"cp367", 3, 5, 3},
      {"csASCII", 3, 7, 2},
      {"ASCII", 3, 5, 1},
      {"ISO_8859-1:1987", 4, 15, 9},
      {"iso-ir-100", 4, 10, 8},
      {"ISO_8859-1", 4, 10, 7},
      {"ISO-8859-1", 4, 10, 6},
      {"latin1", 4, 6, 5},
      {"l1", 4, 2, 4},
      {"IBM819", 4, 6, 3},
      {"CP819", 4, 5, 2},
      {"csISOLatin1", 4, 11, 1},
      {"ISO_8859-2:1987", 5, 15, 7},
      {"iso-ir-101", 5, 10, 6},
      {"ISO_8859-2", 5, 10, 5},
      {"ISO-8859-2", 5, 10, 4},
      {"latin2", 5, 6, 3},
      {"l2", 5, 2, 2},
      {"csISOLatin2", 5, 11, 1},
      {"ISO_8859-3:1988", 6, 15, 7},
      {"iso-ir-109", 6, 10, 6},
      {"ISO_8859-3", 6, 10, 5},
      {"ISO-8859-3", 6, 10, 4},
      {"latin3", 6, 6, 3},
      {"l3", 6, 2, 2},
      {"csISOLatin3", 6, 11, 1},
      {"ISO_8859-4:1988", 7, 15, 7},
      {"iso-ir-110", 7, 10, 6},
      {"ISO_8859-4", 7, 10, 5},
      {"ISO-8859-4", 7, 10, 4},
      {"latin4", 7, 6, 3},
      {"l4", 7, 2, 2},
      {"csISOLatin4", 7, 11, 1},
      {"ISO_8859-5:1988", 8, 15, 6},
      {"iso-ir-144", 8, 10, 5},
      {"ISO_8859-5", 8, 10, 4},
      {"ISO-8859-5", 8, 10, 3},
      {"cyrillic", 8, 8, 2},
      {"csISOLatinCyrillic", 8, 18, 1},
      {"ISO_8859-6:1987", 9, 15, 8},
      {"iso-ir-127", 9, 10, 7},
      {"ISO_8859-6", 9, 10, 6},
      {"ISO-8859-6", 9, 10, 5},
      {"ECMA-114", 9, 8, 4},
      {"ASMO-708", 9, 8, 3},
      {"arabic", 9, 6, 2},
      {"csISOLatinArabic", 9, 16, 1},
      {"ISO_8859-7:1987", 10, 15, 9},
      {"iso-ir-126", 10, 10, 8},
      {"ISO_8859-7", 10, 10, 7},
      {"ISO-8859-7", 10, 10, 6},
      {"ELOT_928", 10, 8, 5},
      {"ECMA-118", 10, 8, 4},
      {"greek", 10, 5, 3},
      {"greek8", 10, 6, 2},
      {"csISOLatinGreek", 10, 15, 1},
      {"ISO_8859-8:1988", 11, 15, 6},
      {"iso-ir-138", 11, 10, 5},
      {"ISO_8859-8", 11, 10, 4},
      {"ISO-8859-8", 11, 10, 3},
      {"hebrew", 11, 6, 2},
      {"csISOLatinHebrew", 11, 16, 1},
      {"ISO_8859-9:1989", 12, 15, 7},
      {"iso-ir-148", 12, 10, 6},
      {"ISO_8859-9", 12, 10, 5},
      {"ISO-8859-9", 12, 10, 4},
      {"latin5", 12, 6, 3},
      {"l5", 12, 2, 2},
      {"csISOLatin5", 12, 11, 1},
      {"ISO-8859-10", 13, 11, 6},
      {"iso-ir-157", 13, 10, 5},
      {"l6", 13, 2, 4},
      {"ISO_8859-10:1992", 13, 16, 3},
      {"csISOLatin6", 13, 11, 2},
      {"latin6", 13, 6, 1},
      {"ISO_6937-2-add", 14, 14, 3},
      {"iso-ir-142", 14, 10, 2},
      {"csISOTextComm", 14, 13, 1},
      {"JIS_X0201", 15, 9, 3},
      {"X0201", 15, 5, 2},
      {"csHalfWidthKatakana", 15, 19, 1},
      {"JIS_Encoding", 16, 12, 2},
      {"csJISEncoding", 16, 13, 1},
      {"Shift_JIS", 17, 9, 3},
      {"MS_Kanji", 17, 8, 2},
      {"csShiftJIS", 17, 10, 1},
      {"Extended_UNIX_Code_Packed_Format_for_Japanese", 18, 45, 3},
      {"csEUCPkdFmtJapanese", 18, 19, 2},
      {"EUC-JP", 18, 6, 1},
      {"Extended_UNIX_Code_Fixed_Width_for_Japanese", 19, 43, 2},
      {"csEUCFixWidJapanese", 19, 19, 1},
      {"BS_4730", 20, 7, 6},
      {"iso-ir-4", 20, 8, 5},
      {"ISO646-GB", 20, 9, 4},
      {"gb", 20, 2, 3},
      {"uk", 20, 2, 2},
      {"csISO4UnitedKingdom", 20, 19, 1},
      {"SEN_850200_C", 21, 12, 5},
      {"iso-ir-11", 21, 9, 4},
      {"ISO646-SE2", 21, 10, 3},
      {"se2", 21, 3, 2},
      {"csISO11SwedishForNames", 21, 22, 1},
      {"IT", 22, 2, 4},
      {"iso-ir-15", 22, 9, 3},
      {"ISO646-IT", 22, 9, 2},
      {"csISO15Italian", 22, 14, 1},
      {"ES", 23, 2, 4},
      {"iso-ir-17", 23, 9, 3},
      {"ISO646-ES", 23, 9, 2},
      {"csISO17Spanish", 23, 14, 1},
      {"DIN_66003", 24, 9, 5},
      {"iso-ir-21", 24, 9, 4},
      {"de", 24, 2, 3},
      {"ISO646-DE", 24, 9, 2},
      {"csISO21German", 24, 13, 1},
      {"NS_4551-1", 25, 9, 6},
      {"iso-ir-60", 25, 9, 5},
      {"ISO646-NO", 25, 9, 4},
      {"no", 25, 2, 3},
      {"csISO60DanishNorwegian", 25, 22, 2},
      {"csISO60Norwegian1", 25, 17, 1},
      {"NF_Z_62-010", 26, 11, 5},
      {"iso-ir-69", 26, 9, 4},
      {"ISO646-FR", 26, 9, 3},
      {"fr", 26, 2, 2},
      {"csISO69French", 26, 13, 1},
      {"ISO-10646-UTF-1", 27, 15, 2},
      {"csISO10646UTF1", 27, 14, 1},
      {"ISO_646.basic:1983", 28, 18, 3},
      {"ref", 28, 3, 2},
      {"csISO646basic1983", 28, 17, 1},
      {"INVARIANT", 29, 9, 2},
      {"csINVARIANT", 29, 11, 1},
      {"ISO_646.irv:1983", 30, 16, 4},
      {"iso-ir-2", 30, 8, 3},
      {"irv", 30, 3, 2},
      {"csISO2IntlRefVersion", 30, 20, 1},
      {"NATS-SEFI", 31, 9, 3},
      {"iso-ir-8-1", 31, 10, 2},
      {"csNATSSEFI", 31, 10, 1},
      {"NATS-SEFI-ADD", 32, 13, 3},
      {"iso-ir-8-2", 32, 10, 2},
      {"csNATSSEFIADD", 32, 13, 1},
      {"SEN_850200_B", 35, 12, 7},
      {"iso-ir-10", 35, 9, 6},
      {"FI", 35, 2, 5},
      {"ISO646-FI", 35, 9, 4},
      {"ISO646-SE", 35, 9, 3},
      {"se", 35, 2, 2},
      {"csISO10Swedish", 35, 14, 1},
      {"KS_C_5601-1987", 36, 14, 6},
      {"iso-ir-149", 36, 10, 5},
      {"KS_C_5601-1989", 36, 14, 4},
      {"KSC_5601", 36, 8, 3},
      {"korean", 36, 6, 2},
      {"csKSC56011987", 36, 13, 1},
      {"ISO-2022-KR", 37, 11, 2},
      {"csISO2022KR", 37, 11, 1},
      {"EUC-KR", 38, 6, 2},
      {"csEUCKR", 38, 7, 1},
      {"ISO-2022-JP", 39, 11, 2},
      {"csISO2022JP", 39, 11, 1},
      {"ISO-2022-JP-2", 40, 13, 2},
      {"csISO2022JP2", 40, 12, 1},
      {"JIS_C6220-1969-jp", 41, 17, 6},
      {"JIS_C6220-1969", 41, 14, 5},
      {"iso-ir-13", 41, 9, 4},
      {"katakana", 41, 8, 3},
      {"x0201-7", 41, 7, 2},
      {"csISO13JISC6220jp", 41, 17, 1},
      {"JIS_C6220-1969-ro", 42, 17, 5},
      {"iso-ir-14", 42, 9, 4},
      {"jp", 42, 2, 3},
      {"ISO646-JP", 42, 9, 2},
      {"csISO14JISC6220ro", 42, 17, 1},
      {"PT", 43, 2, 4},
      {"iso-ir-16", 43, 9, 3},
      {"ISO646-PT", 43, 9, 2},
      {"csISO16Portuguese", 43, 17, 1},
      {"greek7-old", 44, 10, 3},
      {"iso-ir-18", 44, 9, 2},
      {"csISO18Greek7Old", 44, 16, 1},
      {"latin-greek", 45, 11, 3},
      {"iso-ir-19", 45, 9, 2},
      {"csISO19LatinGreek", 45, 17, 1},
      {"NF_Z_62-010_(1973)", 46, 18, 4},
      {"iso-ir-25", 46, 9, 3},
      {"ISO646-FR1", 46, 10, 2},
      {"csISO25French", 46, 13, 1},
      {"Latin-greek-1", 47, 13, 3},
      {"iso-ir-27", 47, 9, 2},
      {"csISO27LatinGreek1", 47, 18, 1},
      {"ISO_5427", 48, 8, 3},
      {"iso-ir-37", 48, 9, 2},
      {"csISO5427Cyrillic", 48, 17, 1},
      {"JIS_C6226-1978", 49, 14, 3},
      {"iso-ir-42", 49, 9, 2},
      {"csISO42JISC62261978", 49, 19, 1},
      {"BS_viewdata", 50, 11, 3},
      {"iso-ir-47", 50, 9, 2},
      {"csISO47BSViewdata", 50, 17, 1},
      {"INIS", 51, 4, 3},
      {"iso-ir-49", 51, 9, 2},
      {"csISO49INIS", 51, 11, 1},
      {"INIS-8", 52, 6, 3},
      {"iso-ir-50", 52, 9, 2},
      {"csISO50INIS8", 52, 12, 1},
      {"INIS-cyrillic", 53, 13, 3},
      {"iso-ir-51", 53, 9, 2},
      {"csISO51INISCyrillic", 53, 19, 1},
      {"ISO_5427:1981", 54, 13, 4},
      {"iso-ir-54", 54, 9, 3},
      {"ISO5427Cyrillic1981", 54, 19, 2},
      {"csISO54271981", 54, 13, 1},
      {"ISO_5428:1980", 55, 13, 3},
      {"iso-ir-55", 55, 9, 2},
      {"csISO5428Greek", 55, 14, 1},
      {"GB_1988-80", 56, 10, 5},
      {"iso-ir-57", 56, 9, 4},
      {"cn", 56, 2, 3},
      {"ISO646-CN", 56, 9, 2},
      {"csISO57GB1988", 56, 13, 1},
      {"GB_2312-80", 57, 10, 4},
      {"iso-ir-58", 57, 9, 3},
      {"chinese", 57, 7, 2},
      {"csISO58GB231280", 57, 15, 1},
      {"NS_4551-2", 58, 9, 5},
      {"ISO646-NO2", 58, 10, 4},
      {"iso-ir-61", 58, 9, 3},
      {"no2", 58, 3, 2},
      {"csISO61Norwegian2", 58, 17, 1},
      {"videotex-suppl", 59, 14, 3},
      {"iso-ir-70", 59, 9, 2},
      {"csISO70VideotexSupp1", 59, 20, 1},
      {"PT2", 60, 3, 4},
      {"iso-ir-84", 60, 9, 3},
      {"ISO646-PT2", 60, 10, 2},
      {"csISO84Portuguese2", 60, 18, 1},
      {"ES2", 61, 3, 4},
      {"iso-ir-85", 61, 9, 3},
      {"ISO646-ES2", 61, 10, 2},
      {"csISO85Spanish2", 61, 15, 1},
      {"MSZ_7795.3", 62, 10, 5},
      {"iso-ir-86", 62, 9, 4},
      {"ISO646-HU", 62, 9, 3},
      {"hu", 62, 2, 2},
      {"csISO86Hungarian", 62, 16, 1},
      {"JIS_C6226-1983", 63, 14, 5},
      {"iso-ir-87", 63, 9, 4},
      {"x0208", 63, 5, 3},
      {"JIS_X0208-1983", 63, 14, 2},
      {"csISO87JISX0208", 63, 15, 1},
      {"greek7", 64, 6, 3},
      {"iso-ir-88", 64, 9, 2},
      {"csISO88Greek7", 64, 13, 1},
      {"ASMO_449", 65, 8, 5},
      {"ISO_9036", 65, 8, 4},
      {"arabic7", 65, 7, 3},
      {"iso-ir-89", 65, 9, 2},
      {"csISO89ASMO449", 65, 14, 1},
      {"iso-ir-90", 66, 9, 2},
      {"csISO90", 66, 7, 1},
      {"JIS_C6229-1984-a", 67, 16, 4},
      {"iso-ir-91", 67, 9, 3},
      {"jp-ocr-a", 67, 8, 2},
      {"csISO91JISC62291984a", 67, 20, 1},
      {"JIS_C6229-1984-b", 68, 16, 5},
      {"iso-ir-92", 68, 9, 4},
      {"ISO646-JP-OCR-B", 68, 15, 3},
      {"jp-ocr-b", 68, 8, 2},
      {"csISO92JISC62991984b", 68, 20, 1},
      {"JIS_C6229-1984-b-add", 69, 20, 4},
      {"iso-ir-93", 69, 9, 3},
      {"jp-ocr-b-add", 69, 12, 2},
      {"csISO93JIS62291984badd", 69, 22, 1},
      {"JIS_C6229-1984-hand", 70, 19, 4},
      {"iso-ir-94", 70, 9, 3},
      {"jp-ocr-hand", 70, 11, 2},
      {"csISO94JIS62291984hand", 70, 22, 1},
      {"JIS_C6229-1984-hand-add", 71, 23, 4},
      {"iso-ir-95", 71, 9, 3},
      {"jp-ocr-hand-add", 71, 15, 2},
      {"csISO95JIS62291984handadd", 71, 25, 1},
      {"JIS_C6229-1984-kana", 72, 19, 3},
      {"iso-ir-96", 72, 9, 2},
      {"csISO96JISC62291984kana", 72, 23, 1},
      {"ISO_2033-1983", 73, 13, 4},
      {"iso-ir-98", 73, 9, 3},
      {"e13b", 73, 4, 2},
      {"csISO2033", 73, 9, 1},
      {"ANSI_X3.110-1983", 74, 16, 5},
      {"iso-ir-99", 74, 9, 4},
      {"CSA_T500-1983", 74, 13, 3},
      {"NAPLPS", 74, 6, 2},
      {"csISO99NAPLPS", 74, 13, 1},
      {"T.61-7bit", 75, 9, 3},
      {"iso-ir-102", 75, 10, 2},
      {"csISO102T617bit", 75, 15, 1},
      {"T.61-8bit", 76, 9, 4},
      {"T.61", 76, 4, 3},
      {"iso-ir-103", 76, 10, 2},
      {"csISO103T618bit", 76, 15, 1},
      {"ECMA-cyrillic", 77, 13, 4},
      {"iso-ir-111", 77, 10, 3},
      {"KOI8-E", 77, 6, 2},
      {"csISO111ECMACyrillic", 77, 20, 1},
      {"CSA_Z243.4-1985-1", 78, 17, 7},
      {"iso-ir-121", 78, 10, 6},
      {"ISO646-CA", 78, 9, 5},
      {"csa7-1", 78, 6, 4},
      {"csa71", 78, 5, 3},
      {"ca", 78, 2, 2},
      {"csISO121Canadian1", 78, 17, 1},
      {"CSA_Z243.4-1985-2", 79, 17, 6},
      {"iso-ir-122", 79, 10, 5},
      {"ISO646-CA2", 79, 10, 4},
      {"csa7-2", 79, 6, 3},
      {"csa72", 79, 5, 2},
      {"csISO122Canadian2", 79, 17, 1},
      {"CSA_Z243.4-1985-gr", 80, 18, 3},
      {"iso-ir-123", 80, 10, 2},
      {"csISO123CSAZ24341985gr", 80, 22, 1},
      {"ISO_8859-6-E", 81, 12, 3},
      {"csISO88596E", 81, 11, 2},
      {"ISO-8859-6-E", 81, 12, 1},
      {"ISO_8859-6-I", 82, 12, 3},
      {"csISO88596I", 82, 11, 2},
      {"ISO-8859-6-I", 82, 12, 1},
      {"T.101-G2", 83, 8, 3},
      {"iso-ir-128", 83, 10, 2},
      {"csISO128T101G2", 83, 14, 1},
      {"ISO_8859-8-E", 84, 12, 3},
      {"csISO88598E", 84, 11, 2},
      {"ISO-8859-8-E", 84, 12, 1},
      {"ISO_8859-8-I", 85, 12, 3},
      {"csISO88598I", 85, 11, 2},
      {"ISO-8859-8-I", 85, 12, 1},
      {"CSN_369103", 86, 10, 3},
      {"iso-ir-139", 86, 10, 2},
      {"csISO139CSN369103", 86, 17, 1},
      {"JUS_I.B1.002", 87, 12, 6},
      {"iso-ir-141", 87, 10, 5},
      {"ISO646-YU", 87, 9, 4},
      {"js", 87, 2, 3},
      {"yu", 87, 2, 2},
      {"csISO141JUSIB1002", 87, 17, 1},
      {"IEC_P27-1", 88, 9, 3},
      {"iso-ir-143", 88, 10, 2},
      {"csISO143IECP271", 88, 15, 1},
      {"JUS_I.B1.003-serb", 89, 17, 4},
      {"iso-ir-146", 89, 10, 3},
      {"serbian", 89, 7, 2},
      {"csISO146Serbian", 89, 15, 1},
      {"JUS_I.B1.003-mac", 90, 16, 4},
      {"macedonian", 90, 10, 3},
      {"iso-ir-147", 90, 10, 2},
      {"csISO147Macedonian", 90, 18, 1},
      {"greek-ccitt", 91, 11, 4},
      {"iso-ir-150", 91, 10, 3},
      {"csISO150", 91, 8, 2},
      {"csISO150GreekCCITT", 91, 18, 1},
      {"NC_NC00-10:81", 92, 13, 5},
      {"cuba", 92, 4, 4},
      {"iso-ir-151", 92, 10, 3},
      {"ISO646-CU", 92, 9, 2},
      {"csISO151Cuba", 92, 12, 1},
      {"ISO_6937-2-25", 93, 13, 3},
      {"iso-ir-152", 93, 10, 2},
      {"csISO6937Add", 93, 12, 1},
      {"GOST_19768-74", 94, 13, 4},
      {"ST_SEV_358-88", 94, 13, 3},
      {"iso-ir-153", 94, 10, 2},
      {"csISO153GOST1976874", 94, 19, 1},
      {"ISO_8859-supp", 95, 13, 4},
      {"iso-ir-154", 95, 10, 3},
      {"latin1-2-5", 95, 10, 2},
      {"csISO8859Supp", 95, 13, 1},
      {"ISO_10367-box", 96, 13, 3},
      {"iso-ir-155", 96, 10, 2},
      {"csISO10367Box", 96, 13, 1},
      {"latin-lap", 97, 9, 4},
      {"lap", 97, 3, 3},
      {"iso-ir-158", 97, 10, 2},
      {"csISO158Lap", 97, 11, 1},
      {"JIS_X0212-1990", 98, 14, 4},
      {"x0212", 98, 5, 3},
      {"iso-ir-159", 98, 10, 2},
      {"csISO159JISX02121990", 98, 20, 1},
      {"DS_2089", 99, 7, 5},
      {"DS2089", 99, 6, 4},
      {"ISO646-DK", 99, 9, 3},
      {"dk", 99, 2, 2},
      {"csISO646Danish", 99, 14, 1},
      {"us-dk", 100, 5, 2},
      {"csUSDK", 100, 6, 1},
      {"dk-us", 101, 5, 2},
      {"csDKUS", 101, 6, 1},
      {"KSC5636", 102, 7, 3},
      {"ISO646-KR", 102, 9, 2},
      {"csKSC5636", 102, 9, 1},
      {"UNICODE-1-1-UTF-7", 103, 17, 2},
      {"csUnicode11UTF7", 103, 15, 1},
      {"ISO-2022-CN", 104, 11, 2},
      {"csISO2022CN", 104, 11, 1},
      {"ISO-2022-CN-EXT", 105, 15, 2},
      {"csISO2022CNEXT", 105, 14, 1},
      {"UTF-8", 106, 5, 2},
      {"csUTF8", 106, 6, 1},
      {"ISO-8859-13", 109, 11, 2},
      {"csISO885913", 109, 11, 1},
      {"ISO-8859-14", 110, 11, 8},
      {"iso-ir-199", 110, 10, 7},
      {"ISO_8859-14:1998", 110, 16, 6},
      {"ISO_8859-14", 110, 11, 5},
      {"latin8", 110, 6, 4},
      {"iso-celtic", 110, 10, 3},
      {"l8", 110, 2, 2},
      {"csISO885914", 110, 11, 1},
      {"ISO-8859-15", 111, 11, 4},
      {"ISO_8859-15", 111, 11, 3},
      {"Latin-9", 111, 7, 2},
      {"csISO885915", 111, 11, 1},
      {"ISO-8859-16", 112, 11, 7},
      {"iso-ir-226", 112, 10, 6},
      {"ISO_8859-16:2001", 112, 16, 5},
      {"ISO_8859-16", 112, 11, 4},
      {"latin10", 112, 7, 3},
      {"l10", 112, 3, 2},
      {"csISO885916", 112, 11, 1},
      {"GBK", 113, 3, 5},
      {"CP936", 113, 5, 4},
      {"MS936", 113, 5, 3},
      {"windows-936", 113, 11, 2},
      {"csGBK", 113, 5, 1},
      {"GB18030", 114, 7, 2},
      {"csGB18030", 114, 9, 1},
      {"OSD_EBCDIC_DF04_15", 115, 18, 2},
      {"csOSDEBCDICDF0415", 115, 17, 1},
      {"OSD_EBCDIC_DF03_IRV", 116, 19, 2},
      {"csOSDEBCDICDF03IRV", 116, 18, 1},
      {"OSD_EBCDIC_DF04_1", 117, 17, 2},
      {"csOSDEBCDICDF041", 117, 16, 1},
      {"ISO-11548-1", 118, 11, 4},
      {"ISO_11548-1", 118, 11, 3},
      {"ISO_TR_11548-1", 118, 14, 2},
      {"csISO115481", 118, 11, 1},
      {"KZ-1048", 119, 7, 4},
      {"STRK1048-2002", 119, 13, 3},
      {"RK1048", 119, 6, 2},
      {"csKZ1048", 119, 8, 1},
      {"ISO-10646-UCS-2", 1000, 15, 2},
      {"csUnicode", 1000, 9, 1},
      {"ISO-10646-UCS-4", 1001, 15, 2},
      {"csUCS4", 1001, 6, 1},
      {"ISO-10646-UCS-Basic", 1002, 19, 2},
      {"csUnicodeASCII", 1002, 14, 1},
      {"ISO-10646-Unicode-Latin1", 1003, 24, 3},
      {"csUnicodeLatin1", 1003, 15, 2},
      {"ISO-10646", 1003, 9, 1},
      {"ISO-10646-J-1", 1004, 13, 2},
      {"csUnicodeJapanese", 1004, 17, 1},
      {"ISO-Unicode-IBM-1261", 1005, 20, 2},
      {"csUnicodeIBM1261", 1005, 16, 1},
      {"ISO-Unicode-IBM-1268", 1006, 20, 2},
      {"csUnicodeIBM1268", 1006, 16, 1},
      {"ISO-Unicode-IBM-1276", 1007, 20, 2},
      {"csUnicodeIBM1276", 1007, 16, 1},
      {"ISO-Unicode-IBM-1264", 1008, 20, 2},
      {"csUnicodeIBM1264", 1008, 16, 1},
      {"ISO-Unicode-IBM-1265", 1009, 20, 2},
      {"csUnicodeIBM1265", 1009, 16, 1},
      {"UNICODE-1-1", 1010, 11, 2},
      {"csUnicode11", 1010, 11, 1},
      {"SCSU", 1011, 4, 2},
      {"csSCSU", 1011, 6, 1},
      {"UTF-7", 1012, 5, 2},
      {"csUTF7", 1012, 6, 1},
      {"UTF-16BE", 1013, 8, 2},
      {"csUTF16BE", 1013, 9, 1},
      {"UTF-16LE", 1014, 8, 2},
      {"csUTF16LE", 1014, 9, 1},
      {"UTF-16", 1015, 6, 2},
      {"csUTF16", 1015, 7, 1},
      {"CESU-8", 1016, 6, 3},
      {"csCESU8", 1016, 7, 2},
      {"csCESU-8", 1016, 8, 1},
      {"UTF-32", 1017, 6, 2},
      {"csUTF32", 1017, 7, 1},
      {"UTF-32BE", 1018, 8, 2},
      {"csUTF32BE", 1018, 9, 1},
      {"UTF-32LE", 1019, 8, 2},
      {"csUTF32LE", 1019, 9, 1},
      {"BOCU-1", 1020, 6, 3},
      {"csBOCU1", 1020, 7, 2},
      {"csBOCU-1", 1020, 8, 1},
      {"UTF-7-IMAP", 1021, 10, 2},
      {"csUTF7IMAP", 1021, 10, 1},
      {"ISO-8859-1-Windows-3.0-Latin-1", 2000, 30, 2},
      {"csWindows30Latin1", 2000, 17, 1},
      {"ISO-8859-1-Windows-3.1-Latin-1", 2001, 30, 2},
      {"csWindows31Latin1", 2001, 17, 1},
      {"ISO-8859-2-Windows-Latin-2", 2002, 26, 2},
      {"csWindows31Latin2", 2002, 17, 1},
      {"ISO-8859-9-Windows-Latin-5", 2003, 26, 2},
      {"csWindows31Latin5", 2003, 17, 1},
      {"hp-roman8", 2004, 9, 4},
      {"roman8", 2004, 6, 3},
      {"r8", 2004, 2, 2},
      {"csHPRoman8", 2004, 10, 1},
      {"Adobe-Standard-Encoding", 2005, 23, 2},
      {"csAdobeStandardEncoding", 2005, 23, 1},
      {"Ventura-US", 2006, 10, 2},
      {"csVenturaUS", 2006, 11, 1},
      {"Ventura-International", 2007, 21, 2},
      {"csVenturaInternational", 2007, 22, 1},
      {"DEC-MCS", 2008, 7, 3},
      {"dec", 2008, 3, 2},
      {"csDECMCS", 2008, 8, 1},
      {"IBM850", 2009, 6, 4},
      {"cp850", 2009, 5, 3},
      {"850", 2009, 3, 2},
      {"csPC850Multilingual", 2009, 19, 1},
      {"IBM852", 2010, 6, 4},
      {"cp852", 2010, 5, 3},
      {"852", 2010, 3, 2},
      {"csPCp852", 2010, 8, 1},
      {"IBM437", 2011, 6, 4},
      {"cp437", 2011, 5, 3},
      {"437", 2011, 3, 2},
      {"csPC8CodePage437", 2011, 16, 1},
      {"PC8-Danish-Norwegian", 2012, 20, 2},
      {"csPC8DanishNorwegian", 2012, 20, 1},
      {"IBM862", 2013, 6, 4},
      {"cp862", 2013, 5, 3},
      {"862", 2013, 3, 2},
      {"csPC862LatinHebrew", 2013, 18, 1},
      {"PC8-Turkish", 2014, 11, 2},
      {"csPC8Turkish", 2014, 12, 1},
      {"IBM-Symbols", 2015, 11, 2},
      {"csIBMSymbols", 2015, 12, 1},
      {"IBM-Thai", 2016, 8, 2},
      {"csIBMThai", 2016, 9, 1},
      {"HP-Legal", 2017, 8, 2},
      {"csHPLegal", 2017, 9, 1},
      {"HP-Pi-font", 2018, 10, 2},
      {"csHPPiFont", 2018, 10, 1},
      {"HP-Math8", 2019, 8, 2},
      {"csHPMath8", 2019, 9, 1},
      {"Adobe-Symbol-Encoding", 2020, 21, 2},
      {"csHPPSMath", 2020, 10, 1},
      {"HP-DeskTop", 2021, 10, 2},
      {"csHPDesktop", 2021, 11, 1},
      {"Ventura-Math", 2022, 12, 2},
      {"csVenturaMath", 2022, 13, 1},
      {"Microsoft-Publishing", 2023, 20, 2},
      {"csMicrosoftPublishing", 2023, 21, 1},
      {"Windows-31J", 2024, 11, 2},
      {"csWindows31J", 2024, 12, 1},
      {"GB2312", 2025, 6, 2},
      {"csGB2312", 2025, 8, 1},
      {"Big5", 2026, 4, 2},
      {"csBig5", 2026, 6, 1},
      {"macintosh", 2027, 9, 3},
      {"mac", 2027, 3, 2},
      {"csMacintosh", 2027, 11, 1},
      {"IBM037", 2028, 6, 7},
      {"cp037", 2028, 5, 6},
      {"ebcdic-cp-us", 2028, 12, 5},
      {"ebcdic-cp-ca", 2028, 12, 4},
      {"ebcdic-cp-wt", 2028, 12, 3},
      {"ebcdic-cp-nl", 2028, 12, 2},
      {"csIBM037", 2028, 8, 1},
      {"IBM038", 2029, 6, 4},
      {"EBCDIC-INT", 2029, 10, 3},
      {"cp038", 2029, 5, 2},
      {"csIBM038", 2029, 8, 1},
      {"IBM273", 2030, 6, 3},
      {"CP273", 2030, 5, 2},
      {"csIBM273", 2030, 8, 1},
      {"IBM274", 2031, 6, 4},
      {"EBCDIC-BE", 2031, 9, 3},
      {"CP274", 2031, 5, 2},
      {"csIBM274", 2031, 8, 1},
      {"IBM275", 2032, 6, 4},
      {"EBCDIC-BR", 2032, 9, 3},
      {"cp275", 2032, 5, 2},
      {"csIBM275", 2032, 8, 1},
      {"IBM277", 2033, 6, 4},
      {"EBCDIC-CP-DK", 2033, 12, 3},
      {"EBCDIC-CP-NO", 2033, 12, 2},
      {"csIBM277", 2033, 8, 1},
      {"IBM278", 2034, 6, 5},
      {"CP278", 2034, 5, 4},
      {"ebcdic-cp-fi", 2034, 12, 3},
      {"ebcdic-cp-se", 2034, 12, 2},
      {"csIBM278", 2034, 8, 1},
      {"IBM280", 2035, 6, 4},
      {"CP280", 2035, 5, 3},
      {"ebcdic-cp-it", 2035, 12, 2},
      {"csIBM280", 2035, 8, 1},
      {"IBM281", 2036, 6, 4},
      {"EBCDIC-JP-E", 2036, 11, 3},
      {"cp281", 2036, 5, 2},
      {"csIBM281", 2036, 8, 1},
      {"IBM284", 2037, 6, 4},
      {"CP284", 2037, 5, 3},
      {"ebcdic-cp-es", 2037, 12, 2},
      {"csIBM284", 2037, 8, 1},
      {"IBM285", 2038, 6, 4},
      {"CP285", 2038, 5, 3},
      {"ebcdic-cp-gb", 2038, 12, 2},
      {"csIBM285", 2038, 8, 1},
      {"IBM290", 2039, 6, 4},
      {"cp290", 2039, 5, 3},
      {"EBCDIC-JP-kana", 2039, 14, 2},
      {"csIBM290", 2039, 8, 1},
      {"IBM297", 2040, 6, 4},
      {"cp297", 2040, 5, 3},
      {"ebcdic-cp-fr", 2040, 12, 2},
      {"csIBM297", 2040, 8, 1},
      {"IBM420", 2041, 6, 4},
      {"cp420", 2041, 5, 3},
      {"ebcdic-cp-ar1", 2041, 13, 2},
      {"csIBM420", 2041, 8, 1},
      {"IBM423", 2042, 6, 4},
      {"cp423", 2042, 5, 3},
      {"ebcdic-cp-gr", 2042, 12, 2},
      {"csIBM423", 2042, 8, 1},
      {"IBM424", 2043, 6, 4},
      {"cp424", 2043, 5, 3},
      {"ebcdic-cp-he", 2043, 12, 2},
      {"csIBM424", 2043, 8, 1},
      {"IBM500", 2044, 6, 5},
      {"CP500", 2044, 5, 4},
      {"ebcdic-cp-be", 2044, 12, 3},
      {"ebcdic-cp-ch", 2044, 12, 2},
      {"csIBM500", 2044, 8, 1},
      {"IBM851", 2045, 6, 4},
      {"cp851", 2045, 5, 3},
      {"851", 2045, 3, 2},
      {"csIBM851", 2045, 8, 1},
      {"IBM855", 2046, 6, 4},
      {"cp855", 2046, 5, 3},
      {"855", 2046, 3, 2},
      {"csIBM855", 2046, 8, 1},
      {"IBM857", 2047, 6, 4},
      {"cp857", 2047, 5, 3},
      {"857", 2047, 3, 2},
      {"csIBM857", 2047, 8, 1},
      {"IBM860", 2048, 6, 4},
      {"cp860", 2048, 5, 3},
      {"860", 2048, 3, 2},
      {"csIBM860", 2048, 8, 1},
      {"IBM861", 2049, 6, 5},
      {"cp861", 2049, 5, 4},
      {"861", 2049, 3, 3},
      {"cp-is", 2049, 5, 2},
      {"csIBM861", 2049, 8, 1},
      {"IBM863", 2050, 6, 4},
      {"cp863", 2050, 5, 3},
      {"863", 2050, 3, 2},
      {"csIBM863", 2050, 8, 1},
      {"IBM864", 2051, 6, 3},
      {"cp864", 2051, 5, 2},
      {"csIBM864", 2051, 8, 1},
      {"IBM865", 2052, 6, 4},
      {"cp865", 2052, 5, 3},
      {"865", 2052, 3, 2},
      {"csIBM865", 2052, 8, 1},
      {"IBM868", 2053, 6, 4},
      {"CP868", 2053, 5, 3},
      {"cp-ar", 2053, 5, 2},
      {"csIBM868", 2053, 8, 1},
      {"IBM869", 2054, 6, 5},
      {"cp869", 2054, 5, 4},
      {"869", 2054, 3, 3},
      {"cp-gr", 2054, 5, 2},
      {"csIBM869", 2054, 8, 1},
      {"IBM870", 2055, 6, 5},
      {"CP870", 2055, 5, 4},
      {"ebcdic-cp-roece", 2055, 15, 3},
      {"ebcdic-cp-yu", 2055, 12, 2},
      {"csIBM870", 2055, 8, 1},
      {"IBM871", 2056, 6, 4},
      {"CP871", 2056, 5, 3},
      {"ebcdic-cp-is", 2056, 12, 2},
      {"csIBM871", 2056, 8, 1},
      {"IBM880", 2057, 6, 4},
      {"cp880", 2057, 5, 3},
      {"EBCDIC-Cyrillic", 2057, 15, 2},
      {"csIBM880", 2057, 8, 1},
      {"IBM891", 2058, 6, 3},
      {"cp891", 2058, 5, 2},
      {"csIBM891", 2058, 8, 1},
      {"IBM903", 2059, 6, 3},
      {"cp903", 2059, 5, 2},
      {"csIBM903", 2059, 8, 1},
      {"IBM904", 2060, 6, 4},
      {"cp904", 2060, 5, 3},
      {"904", 2060, 3, 2},
      {"csIBBM904", 2060, 9, 1},
      {"IBM905", 2061, 6, 4},
      {"CP905", 2061, 5, 3},
      {"ebcdic-cp-tr", 2061, 12, 2},
      {"csIBM905", 2061, 8, 1},
      {"IBM918", 2062, 6, 4},
      {"CP918", 2062, 5, 3},
      {"ebcdic-cp-ar2", 2062, 13, 2},
      {"csIBM918", 2062, 8, 1},
      {"IBM1026", 2063, 7, 3},
      {"CP1026", 2063, 6, 2},
      {"csIBM1026", 2063, 9, 1},
      {"EBCDIC-AT-DE", 2064, 12, 2},
      {"csIBMEBCDICATDE", 2064, 15, 1},
      {"EBCDIC-AT-DE-A", 2065, 14, 2},
      {"csEBCDICATDEA", 2065, 13, 1},
      {"EBCDIC-CA-FR", 2066, 12, 2},
      {"csEBCDICCAFR", 2066, 12, 1},
      {"EBCDIC-DK-NO", 2067, 12, 2},
      {"csEBCDICDKNO", 2067, 12, 1},
      {"EBCDIC-DK-NO-A", 2068, 14, 2},
      {"csEBCDICDKNOA", 2068, 13, 1},
      {"EBCDIC-FI-SE", 2069, 12, 2},
      {"csEBCDICFISE", 2069, 12, 1},
      {"EBCDIC-FI-SE-A", 2070, 14, 2},
      {"csEBCDICFISEA", 2070, 13, 1},
      {"EBCDIC-FR", 2071, 9, 2},
      {"csEBCDICFR", 2071, 10, 1},
      {"EBCDIC-IT", 2072, 9, 2},
      {"csEBCDICIT", 2072, 10, 1},
      {"EBCDIC-PT", 2073, 9, 2},
      {"csEBCDICPT", 2073, 10, 1},
      {"EBCDIC-ES", 2074, 9, 2},
      {"csEBCDICES", 2074, 10, 1},
      {"EBCDIC-ES-A", 2075, 11, 2},
      {"csEBCDICESA", 2075, 11, 1},
      {"EBCDIC-ES-S", 2076, 11, 2},
      {"csEBCDICESS", 2076, 11, 1},
      {"EBCDIC-UK", 2077, 9, 2},
      {"csEBCDICUK", 2077, 10, 1},
      {"EBCDIC-US", 2078, 9, 2},
      {"csEBCDICUS", 2078, 10, 1},
      {"UNKNOWN-8BIT", 2079, 12, 2},
      {"csUnknown8BiT", 2079, 13, 1},
      {"MNEMONIC", 2080, 8, 2},
      {"csMnemonic", 2080, 10, 1},
      {"MNEM", 2081, 4, 2},
      {"csMnem", 2081, 6, 1},
      {"VISCII", 2082, 6, 2},
      {"csVISCII", 2082, 8, 1},
      {"VIQR", 2083, 4, 2},
      {"csVIQR", 2083, 6, 1},
      {"KOI8-R", 2084, 6, 2},
      {"csKOI8R", 2084, 7, 1},
      {"HZ-GB-2312", 2085, 10, 1},
      {"IBM866", 2086, 6, 4},
      {"cp866", 2086, 5, 3},
      {"866", 2086, 3, 2},
      {"csIBM866", 2086, 8, 1},
      {"IBM775", 2087, 6, 3},
      {"cp775", 2087, 5, 2},
      {"csPC775Baltic", 2087, 13, 1},
      {"KOI8-U", 2088, 6, 2},
      {"csKOI8U", 2088, 7, 1},
      {"IBM00858", 2089, 8, 5},
      {"CCSID00858", 2089, 10, 4},
      {"CP00858", 2089, 7, 3},
      {"PC-Multilingual-850+euro", 2089, 24, 2},
      {"csIBM00858", 2089, 10, 1},
      {"IBM00924", 2090, 8, 5},
      {"CCSID00924", 2090, 10, 4},
      {"CP00924", 2090, 7, 3},
      {"ebcdic-Latin9--euro", 2090, 19, 2},
      {"csIBM00924", 2090, 10, 1},
      {"IBM01140", 2091, 8, 5},
      {"CCSID01140", 2091, 10, 4},
      {"CP01140", 2091, 7, 3},
      {"ebcdic-us-37+euro", 2091, 17, 2},
      {"csIBM01140", 2091, 10, 1},
      {"IBM01141", 2092, 8, 5},
      {"CCSID01141", 2092, 10, 4},
      {"CP01141", 2092, 7, 3},
      {"ebcdic-de-273+euro", 2092, 18, 2},
      {"csIBM01141", 2092, 10, 1},
      {"IBM01142", 2093, 8, 6},
      {"CCSID01142", 2093, 10, 5},
      {"CP01142", 2093, 7, 4},
      {"ebcdic-dk-277+euro", 2093, 18, 3},
      {"ebcdic-no-277+euro", 2093, 18, 2},
      {"csIBM01142", 2093, 10, 1},
      {"IBM01143", 2094, 8, 6},
      {"CCSID01143", 2094, 10, 5},
      {"CP01143", 2094, 7, 4},
      {"ebcdic-fi-278+euro", 2094, 18, 3},
      {"ebcdic-se-278+euro", 2094, 18, 2},
      {"csIBM01143", 2094, 10, 1},
      {"IBM01144", 2095, 8, 5},
      {"CCSID01144", 2095, 10, 4},
      {"CP01144", 2095, 7, 3},
      {"ebcdic-it-280+euro", 2095, 18, 2},
      {"csIBM01144", 2095, 10, 1},
      {"IBM01145", 2096, 8, 5},
      {"CCSID01145", 2096, 10, 4},
      {"CP01145", 2096, 7, 3},
      {"ebcdic-es-284+euro", 2096, 18, 2},
      {"csIBM01145", 2096, 10, 1},
      {"IBM01146", 2097, 8, 5},
      {"CCSID01146", 2097, 10, 4},
      {"CP01146", 2097, 7, 3},
      {"ebcdic-gb-285+euro", 2097, 18, 2},
      {"csIBM01146", 2097, 10, 1},
      {"IBM01147", 2098, 8, 5},
      {"CCSID01147", 2098, 10, 4},
      {"CP01147", 2098, 7, 3},
      {"ebcdic-fr-297+euro", 2098, 18, 2},
      {"csIBM01147", 2098, 10, 1},
      {"IBM01148", 2099, 8, 5},
      {"CCSID01148", 2099, 10, 4},
      {"CP01148", 2099, 7, 3},
      {"ebcdic-international-500+euro", 2099, 29, 2},
      {"csIBM01148", 2099, 10, 1},
      {"IBM01149", 2100, 8, 5},
      {"CCSID01149", 2100, 10, 4},
      {"CP01149", 2100, 7, 3},
      {"ebcdic-is-871+euro", 2100, 18, 2},
      {"csIBM01149", 2100, 10, 1},
      {"Big5-HKSCS", 2101, 10, 2},
      {"csBig5HKSCS", 2101, 11, 1},
      {"IBM1047", 2102, 7, 3},
      {"IBM-1047", 2102, 8, 2},
      {"csIBM1047", 2102, 9, 1},
      {"PTCP154", 2103, 7, 5},
      {"csPTCP154", 2103, 9, 4},
      {"PT154", 2103, 5, 3},
      {"CP154", 2103, 5, 2},
      {"Cyrillic-Asian", 2103, 14, 1},
      {"Amiga-1251", 2104, 10, 5},
      {"Ami1251", 2104, 7, 4},
      {"Amiga1251", 2104, 9, 3},
      {"Ami-1251", 2104, 8, 2},
      {"csAmiga1251", 2104, 11, 1},
      {"KOI7-switched", 2105, 13, 2},
      {"csKOI7switched", 2105, 14, 1},
      {"BRF", 2106, 3, 2},
      {"csBRF", 2106, 5, 1},
      {"TSCII", 2107, 5, 2},
      {"csTSCII", 2107, 7, 1},
      {"CP51932", 2108, 7, 2},
      {"csCP51932", 2108, 9, 1},
      {"windows-874", 2109, 11, 2},
      {"cswindows874", 2109, 12, 1},
      {"windows-1250", 2250, 12, 2},
      {"cswindows1250", 2250, 13, 1},
      {"windows-1251", 2251, 12, 2},
      {"cswindows1251", 2251, 13, 1},
      {"windows-1252", 2252, 12, 2},
      {"cswindows1252", 2252, 13, 1},
      {"windows-1253", 2253, 12, 2},
      {"cswindows1253", 2253, 13, 1},
      {"windows-1254", 2254, 12, 2},
      {"cswindows1254", 2254, 13, 1},
      {"windows-1255", 2255, 12, 2},
      {"cswindows1255", 2255, 13, 1},
      {"windows-1256", 2256, 12, 2},
      {"cswindows1256", 2256, 13, 1},
      {"windows-1257", 2257, 12, 2},
      {"cswindows1257", 2257, 13, 1},
      {"windows-1258", 2258, 12, 2},
      {"cswindows1258", 2258, 13, 1},
      {"TIS-620", 2259, 7, 3},
      {"csTIS620", 2259, 8, 2},
      {"ISO-8859-11", 2259, 11, 1},
      {"CP50220", 2260, 7, 2},
      {"csCP50220", 2260, 9, 1}};
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP__TEXT_ENCODING_TE_IMPL_H
