// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H
#define _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#include <__algorithm/copy_n.h>
#include <__algorithm/find.h>
#include <__algorithm/lower_bound.h>
#include <__algorithm/min.h>
#include <__assert>
#include <__functional/hash.h>
#include <__iterator/iterator_traits.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/view_interface.h>
#include <__string/char_traits.h>
#include <__text_encoding/get_locale_encoding.h>
#include <__utility/unreachable.h>
#include <cstdint>
#include <string_view>

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26
_LIBCPP_BEGIN_NAMESPACE_STD

struct text_encoding {
  static constexpr size_t max_name_length = 63;

private:
  using __id_rep _LIBCPP_NODEBUG = int_least32_t;
  struct __encoding_data {
    const char* __name_;
    __id_rep __mib_rep_;
    uint_least32_t __name_size_;

    friend constexpr bool operator==(const __encoding_data& __e, const __encoding_data& __other) noexcept {
      return __e.__mib_rep_ == __other.__mib_rep_ ||
             __comp_name(string_view(__e.__name_, __e.__name_size_), string_view(__other.__name_, __e.__name_size_));
    }

    friend constexpr bool operator<(const __encoding_data& __e, const __id_rep __i) noexcept {
      return __e.__mib_rep_ < __i;
    }

    friend constexpr bool operator==(const __encoding_data& __e, std::string_view __name) noexcept {
      return __comp_name(__name, string_view(__e.__name_, __e.__name_size_));
    }
  };

public:
  enum class id : __id_rep {
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

  using enum id;

  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding() = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit text_encoding(string_view __enc) noexcept
      : __encoding_rep_(__find_encoding_data(__enc)) {
    __enc.copy(__name_, max_name_length, 0);
  }
  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding(id __i) noexcept : __encoding_rep_(__find_encoding_data_by_id(__i)) {
    if (__encoding_rep_->__name_[0] != '\0')
      std::copy_n(__encoding_rep_->__name_, std::char_traits<char>::length(__encoding_rep_->__name_), __name_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr id mib() const noexcept { return id(__encoding_rep_->__mib_rep_); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const char* name() const noexcept { return __name_; }

  // [text.encoding.aliases], class text_encoding::aliases_view
  struct aliases_view : ranges::view_interface<aliases_view> {
    constexpr aliases_view() = default;
    constexpr aliases_view(const __encoding_data* __d) : __view_data_(__d) {}
    struct __end_sentinel {};
    struct __iterator {
      using value_type      = const char*;
      using reference       = const char*;
      using difference_type = ptrdiff_t;

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator() noexcept = default;

      _LIBCPP_HIDE_FROM_ABI constexpr value_type operator*() const {
        _LIBCPP_ASSERT(__can_dereference(), "Dereferencing invalid aliases_view iterator!");
        return __data_->__name_;
      }

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
        _LIBCPP_ASSERT(__other.__mib_rep_ == __mib_rep_, "Subtracting ranges of two different text encodings!");
        return __mib_rep_ - __other.__mib_rep_;
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

      // Check if going past the encoding data list array and if the new index has the same id, if not then
      // replace it with a sentinel "out-of-bounds" iterator.
      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n) {
        if (__data_) {
          if (__n > 0) {
            if ((__data_ + __n) < std::end(__text_encoding_data) && __data_[__n - 1].__mib_rep_ == __mib_rep_)
              __data_ += __n;
            else
              *this = __iterator{};
          } else if (__n < 0) {
            if ((__data_ + __n) > __text_encoding_data && __data_[__n].__mib_rep_ == __mib_rep_)
              __data_ += __n;
            else
              *this = __iterator{};
          }
        }
        return *this;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n) { return operator+=(-__n); }

      _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const __iterator& __it) const {
        return __data_ == __it.__data_ && __it.__mib_rep_ == __mib_rep_;
      }

      _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(__end_sentinel) const { return !__can_dereference(); }

      _LIBCPP_HIDE_FROM_ABI constexpr auto operator<=>(__iterator __it) const { return __data_ <=> __it.__data_; }

    private:
      friend struct text_encoding;

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator(const __encoding_data* __enc_d) noexcept
          : __data_(__enc_d), __mib_rep_(__enc_d ? __enc_d->__mib_rep_ : 0) {}

      _LIBCPP_HIDE_FROM_ABI constexpr bool __can_dereference() const {
        return __data_ && __data_->__mib_rep_ == __mib_rep_;
      }

      // default iterator is a sentinel
      const __encoding_data* __data_ = nullptr;
      __id_rep __mib_rep_            = 0;
    };

    constexpr __iterator begin() const { return __iterator{__view_data_}; }
    constexpr __end_sentinel end() const { return __end_sentinel{}; }

  private:
    const __encoding_data* __view_data_ = nullptr;
  };

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr aliases_view aliases() const noexcept {
    if (!__encoding_rep_->__name_[0])
      return aliases_view(nullptr);
    return aliases_view(__encoding_rep_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __a, const text_encoding& __b) noexcept {
    if (__a.mib() == id::other && __b.mib() == id::other)
      return __comp_name(__a.__name_, __b.__name_);

    return __a.mib() == __b.mib();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __encoding, id __i) noexcept {
    return __encoding.mib() == __i;
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static consteval text_encoding literal() noexcept {
    // TODO: Remove this branch once we have __GNUC_EXECUTION_CHARSET_NAME or __clang_literal_encoding__ unconditionally
#  ifdef __GNUC_EXECUTION_CHARSET_NAME
    return text_encoding(__GNUC_EXECUTION_CHARSET_NAME);
#  elif defined(__clang_literal_encoding__)
    return text_encoding(__clang_literal_encoding__);
#  else
    return text_encoding();
#  endif
  }

#  if _LIBCPP_HAS_LOCALIZATION
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static text_encoding environment() {
    return text_encoding(__get_locale_encoding(""));
  }

  template <id __i>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is() {
    return environment() == __i;
  }
#  else
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static text_encoding environment() = delete;
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is()       = delete;
#  endif // _LIBCPP_HAS_LOCALIZATION

private:
  _LIBCPP_HIDE_FROM_ABI static constexpr bool __comp_name(string_view __a, string_view __b) {
    if (__a.empty() || __b.empty()) {
      return false;
    }

    // map any non-alphanumeric character to 255, skip prefix 0s, else get tolower(__n)
    auto __map_char = [](char __n, bool& __in_number) -> unsigned char {
      auto __to_lower = [](char __n) -> char { return (__n >= 'A' && __n <= 'Z') ? __n + ('a' - 'A') : __n; };
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
      while ((__a_val = __map_char(*__a_ptr, __a_in_number)) == 255 && __a_ptr != __a.end())
        __a_ptr++;
      while ((__b_val = __map_char(*__b_ptr, __b_in_number)) == 255 && __b_ptr != __b.end())
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

  _LIBCPP_HIDE_FROM_ABI static constexpr const __encoding_data* __find_encoding_data(string_view __a) {
    _LIBCPP_ASSERT(__a.size() <= max_name_length, "Passing encoding name longer than max_name_length!");
    auto __data_ptr = __text_encoding_data + 2, __data_last = std::end(__text_encoding_data);
    auto __found_data = std::find(__data_ptr, __data_last, __a);

    if (__found_data == __data_last) {
      return __text_encoding_data; // other
    }

    while (__found_data[-1].__mib_rep_ == __found_data->__mib_rep_) {
      __found_data--;
    }

    return __found_data;
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr const __encoding_data* __find_encoding_data_by_id(id __i) {
    _LIBCPP_ASSERT(__i >= id::other && __i <= id::CP50220 && __id_rep(__i) != 33 && __id_rep(__i) != 34,
                   "Passing invalid id to text_encoding constructor!");
    auto __found = std::lower_bound(std::begin(__text_encoding_data), std::end(__text_encoding_data), __id_rep(__i));
    return __found != std::end(__text_encoding_data)
             ? __found
             : __text_encoding_data + 1; // unknown, should be unreachable
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr __encoding_data __text_encoding_data[] = {
      {"", 1, 0},
      {"", 2, 0},
      {"ANSI_X3.4-1968", 3, 14},
      {"ANSI_X3.4-1986", 3, 14},
      {"IBM367", 3, 6},
      {"ISO646-US", 3, 9},
      {"ISO_646.irv:1991", 3, 16},
      {"cp367", 3, 5},
      {"csASCII", 3, 7},
      {"iso-ir-6", 3, 8},
      {"us", 3, 2},
      {"ISO-8859-1", 4, 10},
      {"ISO_8859-1:1987", 4, 15},
      {"CP819", 4, 5},
      {"IBM819", 4, 6},
      {"ISO_8859-1", 4, 10},
      {"csISOLatin1", 4, 11},
      {"iso-ir-100", 4, 10},
      {"l1", 4, 2},
      {"latin1", 4, 6},
      {"ISO-8859-2", 5, 10},
      {"ISO_8859-2:1987", 5, 15},
      {"ISO_8859-2", 5, 10},
      {"csISOLatin2", 5, 11},
      {"iso-ir-101", 5, 10},
      {"l2", 5, 2},
      {"latin2", 5, 6},
      {"ISO-8859-3", 6, 10},
      {"ISO_8859-3:1988", 6, 15},
      {"ISO_8859-3", 6, 10},
      {"csISOLatin3", 6, 11},
      {"iso-ir-109", 6, 10},
      {"l3", 6, 2},
      {"latin3", 6, 6},
      {"ISO-8859-4", 7, 10},
      {"ISO_8859-4:1988", 7, 15},
      {"ISO_8859-4", 7, 10},
      {"csISOLatin4", 7, 11},
      {"iso-ir-110", 7, 10},
      {"l4", 7, 2},
      {"latin4", 7, 6},
      {"ISO-8859-5", 8, 10},
      {"ISO_8859-5:1988", 8, 15},
      {"ISO_8859-5", 8, 10},
      {"csISOLatinCyrillic", 8, 18},
      {"cyrillic", 8, 8},
      {"iso-ir-144", 8, 10},
      {"ISO-8859-6", 9, 10},
      {"ISO_8859-6:1987", 9, 15},
      {"ASMO-708", 9, 8},
      {"ECMA-114", 9, 8},
      {"ISO_8859-6", 9, 10},
      {"arabic", 9, 6},
      {"csISOLatinArabic", 9, 16},
      {"iso-ir-127", 9, 10},
      {"ISO-8859-7", 10, 10},
      {"ISO_8859-7:1987", 10, 15},
      {"ECMA-118", 10, 8},
      {"ELOT_928", 10, 8},
      {"ISO_8859-7", 10, 10},
      {"csISOLatinGreek", 10, 15},
      {"greek", 10, 5},
      {"greek8", 10, 6},
      {"iso-ir-126", 10, 10},
      {"ISO-8859-8", 11, 10},
      {"ISO_8859-8:1988", 11, 15},
      {"ISO_8859-8", 11, 10},
      {"csISOLatinHebrew", 11, 16},
      {"hebrew", 11, 6},
      {"iso-ir-138", 11, 10},
      {"ISO-8859-9", 12, 10},
      {"ISO_8859-9:1989", 12, 15},
      {"ISO_8859-9", 12, 10},
      {"csISOLatin5", 12, 11},
      {"iso-ir-148", 12, 10},
      {"l5", 12, 2},
      {"latin5", 12, 6},
      {"ISO-8859-10", 13, 11},
      {"ISO_8859-10:1992", 13, 16},
      {"csISOLatin6", 13, 11},
      {"iso-ir-157", 13, 10},
      {"l6", 13, 2},
      {"latin6", 13, 6},
      {"ISO_6937-2-add", 14, 14},
      {"csISOTextComm", 14, 13},
      {"iso-ir-142", 14, 10},
      {"JIS_X0201", 15, 9},
      {"X0201", 15, 5},
      {"csHalfWidthKatakana", 15, 19},
      {"JIS_Encoding", 16, 12},
      {"csJISEncoding", 16, 13},
      {"Shift_JIS", 17, 9},
      {"MS_Kanji", 17, 8},
      {"csShiftJIS", 17, 10},
      {"EUC-JP", 18, 6},
      {"Extended_UNIX_Code_Packed_Format_for_Japanese", 18, 45},
      {"csEUCPkdFmtJapanese", 18, 19},
      {"Extended_UNIX_Code_Fixed_Width_for_Japanese", 19, 43},
      {"csEUCFixWidJapanese", 19, 19},
      {"BS_4730", 20, 7},
      {"ISO646-GB", 20, 9},
      {"csISO4UnitedKingdom", 20, 19},
      {"gb", 20, 2},
      {"iso-ir-4", 20, 8},
      {"uk", 20, 2},
      {"SEN_850200_C", 21, 12},
      {"ISO646-SE2", 21, 10},
      {"csISO11SwedishForNames", 21, 22},
      {"iso-ir-11", 21, 9},
      {"se2", 21, 3},
      {"IT", 22, 2},
      {"ISO646-IT", 22, 9},
      {"csISO15Italian", 22, 14},
      {"iso-ir-15", 22, 9},
      {"ES", 23, 2},
      {"ISO646-ES", 23, 9},
      {"csISO17Spanish", 23, 14},
      {"iso-ir-17", 23, 9},
      {"DIN_66003", 24, 9},
      {"ISO646-DE", 24, 9},
      {"csISO21German", 24, 13},
      {"de", 24, 2},
      {"iso-ir-21", 24, 9},
      {"NS_4551-1", 25, 9},
      {"ISO646-NO", 25, 9},
      {"csISO60DanishNorwegian", 25, 22},
      {"csISO60Norwegian1", 25, 17},
      {"iso-ir-60", 25, 9},
      {"no", 25, 2},
      {"NF_Z_62-010", 26, 11},
      {"ISO646-FR", 26, 9},
      {"csISO69French", 26, 13},
      {"fr", 26, 2},
      {"iso-ir-69", 26, 9},
      {"ISO-10646-UTF-1", 27, 15},
      {"csISO10646UTF1", 27, 14},
      {"ISO_646.basic:1983", 28, 18},
      {"csISO646basic1983", 28, 17},
      {"ref", 28, 3},
      {"INVARIANT", 29, 9},
      {"csINVARIANT", 29, 11},
      {"ISO_646.irv:1983", 30, 16},
      {"csISO2IntlRefVersion", 30, 20},
      {"irv", 30, 3},
      {"iso-ir-2", 30, 8},
      {"NATS-SEFI", 31, 9},
      {"csNATSSEFI", 31, 10},
      {"iso-ir-8-1", 31, 10},
      {"NATS-SEFI-ADD", 32, 13},
      {"csNATSSEFIADD", 32, 13},
      {"iso-ir-8-2", 32, 10},
      {"SEN_850200_B", 35, 12},
      {"FI", 35, 2},
      {"ISO646-FI", 35, 9},
      {"ISO646-SE", 35, 9},
      {"csISO10Swedish", 35, 14},
      {"iso-ir-10", 35, 9},
      {"se", 35, 2},
      {"KS_C_5601-1987", 36, 14},
      {"KSC_5601", 36, 8},
      {"KS_C_5601-1989", 36, 14},
      {"csKSC56011987", 36, 13},
      {"iso-ir-149", 36, 10},
      {"korean", 36, 6},
      {"ISO-2022-KR", 37, 11},
      {"csISO2022KR", 37, 11},
      {"EUC-KR", 38, 6},
      {"csEUCKR", 38, 7},
      {"ISO-2022-JP", 39, 11},
      {"csISO2022JP", 39, 11},
      {"ISO-2022-JP-2", 40, 13},
      {"csISO2022JP2", 40, 12},
      {"JIS_C6220-1969-jp", 41, 17},
      {"JIS_C6220-1969", 41, 14},
      {"csISO13JISC6220jp", 41, 17},
      {"iso-ir-13", 41, 9},
      {"katakana", 41, 8},
      {"x0201-7", 41, 7},
      {"JIS_C6220-1969-ro", 42, 17},
      {"ISO646-JP", 42, 9},
      {"csISO14JISC6220ro", 42, 17},
      {"iso-ir-14", 42, 9},
      {"jp", 42, 2},
      {"PT", 43, 2},
      {"ISO646-PT", 43, 9},
      {"csISO16Portuguese", 43, 17},
      {"iso-ir-16", 43, 9},
      {"greek7-old", 44, 10},
      {"csISO18Greek7Old", 44, 16},
      {"iso-ir-18", 44, 9},
      {"latin-greek", 45, 11},
      {"csISO19LatinGreek", 45, 17},
      {"iso-ir-19", 45, 9},
      {"NF_Z_62-010_(1973)", 46, 18},
      {"ISO646-FR1", 46, 10},
      {"csISO25French", 46, 13},
      {"iso-ir-25", 46, 9},
      {"Latin-greek-1", 47, 13},
      {"csISO27LatinGreek1", 47, 18},
      {"iso-ir-27", 47, 9},
      {"ISO_5427", 48, 8},
      {"csISO5427Cyrillic", 48, 17},
      {"iso-ir-37", 48, 9},
      {"JIS_C6226-1978", 49, 14},
      {"csISO42JISC62261978", 49, 19},
      {"iso-ir-42", 49, 9},
      {"BS_viewdata", 50, 11},
      {"csISO47BSViewdata", 50, 17},
      {"iso-ir-47", 50, 9},
      {"INIS", 51, 4},
      {"csISO49INIS", 51, 11},
      {"iso-ir-49", 51, 9},
      {"INIS-8", 52, 6},
      {"csISO50INIS8", 52, 12},
      {"iso-ir-50", 52, 9},
      {"INIS-cyrillic", 53, 13},
      {"csISO51INISCyrillic", 53, 19},
      {"iso-ir-51", 53, 9},
      {"ISO_5427:1981", 54, 13},
      {"ISO5427Cyrillic1981", 54, 19},
      {"csISO54271981", 54, 13},
      {"iso-ir-54", 54, 9},
      {"ISO_5428:1980", 55, 13},
      {"csISO5428Greek", 55, 14},
      {"iso-ir-55", 55, 9},
      {"GB_1988-80", 56, 10},
      {"ISO646-CN", 56, 9},
      {"cn", 56, 2},
      {"csISO57GB1988", 56, 13},
      {"iso-ir-57", 56, 9},
      {"GB_2312-80", 57, 10},
      {"chinese", 57, 7},
      {"csISO58GB231280", 57, 15},
      {"iso-ir-58", 57, 9},
      {"NS_4551-2", 58, 9},
      {"ISO646-NO2", 58, 10},
      {"csISO61Norwegian2", 58, 17},
      {"iso-ir-61", 58, 9},
      {"no2", 58, 3},
      {"videotex-suppl", 59, 14},
      {"csISO70VideotexSupp1", 59, 20},
      {"iso-ir-70", 59, 9},
      {"PT2", 60, 3},
      {"ISO646-PT2", 60, 10},
      {"csISO84Portuguese2", 60, 18},
      {"iso-ir-84", 60, 9},
      {"ES2", 61, 3},
      {"ISO646-ES2", 61, 10},
      {"csISO85Spanish2", 61, 15},
      {"iso-ir-85", 61, 9},
      {"MSZ_7795.3", 62, 10},
      {"ISO646-HU", 62, 9},
      {"csISO86Hungarian", 62, 16},
      {"hu", 62, 2},
      {"iso-ir-86", 62, 9},
      {"JIS_C6226-1983", 63, 14},
      {"JIS_X0208-1983", 63, 14},
      {"csISO87JISX0208", 63, 15},
      {"iso-ir-87", 63, 9},
      {"x0208", 63, 5},
      {"greek7", 64, 6},
      {"csISO88Greek7", 64, 13},
      {"iso-ir-88", 64, 9},
      {"ASMO_449", 65, 8},
      {"ISO_9036", 65, 8},
      {"arabic7", 65, 7},
      {"csISO89ASMO449", 65, 14},
      {"iso-ir-89", 65, 9},
      {"iso-ir-90", 66, 9},
      {"csISO90", 66, 7},
      {"JIS_C6229-1984-a", 67, 16},
      {"csISO91JISC62291984a", 67, 20},
      {"iso-ir-91", 67, 9},
      {"jp-ocr-a", 67, 8},
      {"JIS_C6229-1984-b", 68, 16},
      {"ISO646-JP-OCR-B", 68, 15},
      {"csISO92JISC62991984b", 68, 20},
      {"iso-ir-92", 68, 9},
      {"jp-ocr-b", 68, 8},
      {"JIS_C6229-1984-b-add", 69, 20},
      {"csISO93JIS62291984badd", 69, 22},
      {"iso-ir-93", 69, 9},
      {"jp-ocr-b-add", 69, 12},
      {"JIS_C6229-1984-hand", 70, 19},
      {"csISO94JIS62291984hand", 70, 22},
      {"iso-ir-94", 70, 9},
      {"jp-ocr-hand", 70, 11},
      {"JIS_C6229-1984-hand-add", 71, 23},
      {"csISO95JIS62291984handadd", 71, 25},
      {"iso-ir-95", 71, 9},
      {"jp-ocr-hand-add", 71, 15},
      {"JIS_C6229-1984-kana", 72, 19},
      {"csISO96JISC62291984kana", 72, 23},
      {"iso-ir-96", 72, 9},
      {"ISO_2033-1983", 73, 13},
      {"csISO2033", 73, 9},
      {"e13b", 73, 4},
      {"iso-ir-98", 73, 9},
      {"ANSI_X3.110-1983", 74, 16},
      {"CSA_T500-1983", 74, 13},
      {"NAPLPS", 74, 6},
      {"csISO99NAPLPS", 74, 13},
      {"iso-ir-99", 74, 9},
      {"T.61-7bit", 75, 9},
      {"csISO102T617bit", 75, 15},
      {"iso-ir-102", 75, 10},
      {"T.61-8bit", 76, 9},
      {"T.61", 76, 4},
      {"csISO103T618bit", 76, 15},
      {"iso-ir-103", 76, 10},
      {"ECMA-cyrillic", 77, 13},
      {"KOI8-E", 77, 6},
      {"csISO111ECMACyrillic", 77, 20},
      {"iso-ir-111", 77, 10},
      {"CSA_Z243.4-1985-1", 78, 17},
      {"ISO646-CA", 78, 9},
      {"ca", 78, 2},
      {"csISO121Canadian1", 78, 17},
      {"csa7-1", 78, 6},
      {"csa71", 78, 5},
      {"iso-ir-121", 78, 10},
      {"CSA_Z243.4-1985-2", 79, 17},
      {"ISO646-CA2", 79, 10},
      {"csISO122Canadian2", 79, 17},
      {"csa7-2", 79, 6},
      {"csa72", 79, 5},
      {"iso-ir-122", 79, 10},
      {"CSA_Z243.4-1985-gr", 80, 18},
      {"csISO123CSAZ24341985gr", 80, 22},
      {"iso-ir-123", 80, 10},
      {"ISO-8859-6-E", 81, 12},
      {"ISO_8859-6-E", 81, 12},
      {"csISO88596E", 81, 11},
      {"ISO-8859-6-I", 82, 12},
      {"ISO_8859-6-I", 82, 12},
      {"csISO88596I", 82, 11},
      {"T.101-G2", 83, 8},
      {"csISO128T101G2", 83, 14},
      {"iso-ir-128", 83, 10},
      {"ISO-8859-8-E", 84, 12},
      {"ISO_8859-8-E", 84, 12},
      {"csISO88598E", 84, 11},
      {"ISO-8859-8-I", 85, 12},
      {"ISO_8859-8-I", 85, 12},
      {"csISO88598I", 85, 11},
      {"CSN_369103", 86, 10},
      {"csISO139CSN369103", 86, 17},
      {"iso-ir-139", 86, 10},
      {"JUS_I.B1.002", 87, 12},
      {"ISO646-YU", 87, 9},
      {"csISO141JUSIB1002", 87, 17},
      {"iso-ir-141", 87, 10},
      {"js", 87, 2},
      {"yu", 87, 2},
      {"IEC_P27-1", 88, 9},
      {"csISO143IECP271", 88, 15},
      {"iso-ir-143", 88, 10},
      {"JUS_I.B1.003-serb", 89, 17},
      {"csISO146Serbian", 89, 15},
      {"iso-ir-146", 89, 10},
      {"serbian", 89, 7},
      {"JUS_I.B1.003-mac", 90, 16},
      {"csISO147Macedonian", 90, 18},
      {"iso-ir-147", 90, 10},
      {"macedonian", 90, 10},
      {"greek-ccitt", 91, 11},
      {"csISO150", 91, 8},
      {"csISO150GreekCCITT", 91, 18},
      {"iso-ir-150", 91, 10},
      {"NC_NC00-10:81", 92, 13},
      {"ISO646-CU", 92, 9},
      {"csISO151Cuba", 92, 12},
      {"cuba", 92, 4},
      {"iso-ir-151", 92, 10},
      {"ISO_6937-2-25", 93, 13},
      {"csISO6937Add", 93, 12},
      {"iso-ir-152", 93, 10},
      {"GOST_19768-74", 94, 13},
      {"ST_SEV_358-88", 94, 13},
      {"csISO153GOST1976874", 94, 19},
      {"iso-ir-153", 94, 10},
      {"ISO_8859-supp", 95, 13},
      {"csISO8859Supp", 95, 13},
      {"iso-ir-154", 95, 10},
      {"latin1-2-5", 95, 10},
      {"ISO_10367-box", 96, 13},
      {"csISO10367Box", 96, 13},
      {"iso-ir-155", 96, 10},
      {"latin-lap", 97, 9},
      {"csISO158Lap", 97, 11},
      {"iso-ir-158", 97, 10},
      {"lap", 97, 3},
      {"JIS_X0212-1990", 98, 14},
      {"csISO159JISX02121990", 98, 20},
      {"iso-ir-159", 98, 10},
      {"x0212", 98, 5},
      {"DS_2089", 99, 7},
      {"DS2089", 99, 6},
      {"ISO646-DK", 99, 9},
      {"csISO646Danish", 99, 14},
      {"dk", 99, 2},
      {"us-dk", 100, 5},
      {"csUSDK", 100, 6},
      {"dk-us", 101, 5},
      {"csDKUS", 101, 6},
      {"KSC5636", 102, 7},
      {"ISO646-KR", 102, 9},
      {"csKSC5636", 102, 9},
      {"UNICODE-1-1-UTF-7", 103, 17},
      {"csUnicode11UTF7", 103, 15},
      {"ISO-2022-CN", 104, 11},
      {"csISO2022CN", 104, 11},
      {"ISO-2022-CN-EXT", 105, 15},
      {"csISO2022CNEXT", 105, 14},
      {"UTF-8", 106, 5},
      {"csUTF8", 106, 6},
      {"ISO-8859-13", 109, 11},
      {"csISO885913", 109, 11},
      {"ISO-8859-14", 110, 11},
      {"ISO_8859-14", 110, 11},
      {"ISO_8859-14:1998", 110, 16},
      {"csISO885914", 110, 11},
      {"iso-celtic", 110, 10},
      {"iso-ir-199", 110, 10},
      {"l8", 110, 2},
      {"latin8", 110, 6},
      {"ISO-8859-15", 111, 11},
      {"ISO_8859-15", 111, 11},
      {"Latin-9", 111, 7},
      {"csISO885915", 111, 11},
      {"ISO-8859-16", 112, 11},
      {"ISO_8859-16", 112, 11},
      {"ISO_8859-16:2001", 112, 16},
      {"csISO885916", 112, 11},
      {"iso-ir-226", 112, 10},
      {"l10", 112, 3},
      {"latin10", 112, 7},
      {"GBK", 113, 3},
      {"CP936", 113, 5},
      {"MS936", 113, 5},
      {"csGBK", 113, 5},
      {"windows-936", 113, 11},
      {"GB18030", 114, 7},
      {"csGB18030", 114, 9},
      {"OSD_EBCDIC_DF04_15", 115, 18},
      {"csOSDEBCDICDF0415", 115, 17},
      {"OSD_EBCDIC_DF03_IRV", 116, 19},
      {"csOSDEBCDICDF03IRV", 116, 18},
      {"OSD_EBCDIC_DF04_1", 117, 17},
      {"csOSDEBCDICDF041", 117, 16},
      {"ISO-11548-1", 118, 11},
      {"ISO_11548-1", 118, 11},
      {"ISO_TR_11548-1", 118, 14},
      {"csISO115481", 118, 11},
      {"KZ-1048", 119, 7},
      {"RK1048", 119, 6},
      {"STRK1048-2002", 119, 13},
      {"csKZ1048", 119, 8},
      {"ISO-10646-UCS-2", 1000, 15},
      {"csUnicode", 1000, 9},
      {"ISO-10646-UCS-4", 1001, 15},
      {"csUCS4", 1001, 6},
      {"ISO-10646-UCS-Basic", 1002, 19},
      {"csUnicodeASCII", 1002, 14},
      {"ISO-10646-Unicode-Latin1", 1003, 24},
      {"ISO-10646", 1003, 9},
      {"csUnicodeLatin1", 1003, 15},
      {"ISO-10646-J-1", 1004, 13},
      {"csUnicodeJapanese", 1004, 17},
      {"ISO-Unicode-IBM-1261", 1005, 20},
      {"csUnicodeIBM1261", 1005, 16},
      {"ISO-Unicode-IBM-1268", 1006, 20},
      {"csUnicodeIBM1268", 1006, 16},
      {"ISO-Unicode-IBM-1276", 1007, 20},
      {"csUnicodeIBM1276", 1007, 16},
      {"ISO-Unicode-IBM-1264", 1008, 20},
      {"csUnicodeIBM1264", 1008, 16},
      {"ISO-Unicode-IBM-1265", 1009, 20},
      {"csUnicodeIBM1265", 1009, 16},
      {"UNICODE-1-1", 1010, 11},
      {"csUnicode11", 1010, 11},
      {"SCSU", 1011, 4},
      {"csSCSU", 1011, 6},
      {"UTF-7", 1012, 5},
      {"csUTF7", 1012, 6},
      {"UTF-16BE", 1013, 8},
      {"csUTF16BE", 1013, 9},
      {"UTF-16LE", 1014, 8},
      {"csUTF16LE", 1014, 9},
      {"UTF-16", 1015, 6},
      {"csUTF16", 1015, 7},
      {"CESU-8", 1016, 6},
      {"csCESU-8", 1016, 8},
      {"csCESU8", 1016, 7},
      {"UTF-32", 1017, 6},
      {"csUTF32", 1017, 7},
      {"UTF-32BE", 1018, 8},
      {"csUTF32BE", 1018, 9},
      {"UTF-32LE", 1019, 8},
      {"csUTF32LE", 1019, 9},
      {"BOCU-1", 1020, 6},
      {"csBOCU-1", 1020, 8},
      {"csBOCU1", 1020, 7},
      {"UTF-7-IMAP", 1021, 10},
      {"csUTF7IMAP", 1021, 10},
      {"ISO-8859-1-Windows-3.0-Latin-1", 2000, 30},
      {"csWindows30Latin1", 2000, 17},
      {"ISO-8859-1-Windows-3.1-Latin-1", 2001, 30},
      {"csWindows31Latin1", 2001, 17},
      {"ISO-8859-2-Windows-Latin-2", 2002, 26},
      {"csWindows31Latin2", 2002, 17},
      {"ISO-8859-9-Windows-Latin-5", 2003, 26},
      {"csWindows31Latin5", 2003, 17},
      {"hp-roman8", 2004, 9},
      {"csHPRoman8", 2004, 10},
      {"r8", 2004, 2},
      {"roman8", 2004, 6},
      {"Adobe-Standard-Encoding", 2005, 23},
      {"csAdobeStandardEncoding", 2005, 23},
      {"Ventura-US", 2006, 10},
      {"csVenturaUS", 2006, 11},
      {"Ventura-International", 2007, 21},
      {"csVenturaInternational", 2007, 22},
      {"DEC-MCS", 2008, 7},
      {"csDECMCS", 2008, 8},
      {"dec", 2008, 3},
      {"IBM850", 2009, 6},
      {"850", 2009, 3},
      {"cp850", 2009, 5},
      {"csPC850Multilingual", 2009, 19},
      {"IBM852", 2010, 6},
      {"852", 2010, 3},
      {"cp852", 2010, 5},
      {"csPCp852", 2010, 8},
      {"IBM437", 2011, 6},
      {"437", 2011, 3},
      {"cp437", 2011, 5},
      {"csPC8CodePage437", 2011, 16},
      {"PC8-Danish-Norwegian", 2012, 20},
      {"csPC8DanishNorwegian", 2012, 20},
      {"IBM862", 2013, 6},
      {"862", 2013, 3},
      {"cp862", 2013, 5},
      {"csPC862LatinHebrew", 2013, 18},
      {"PC8-Turkish", 2014, 11},
      {"csPC8Turkish", 2014, 12},
      {"IBM-Symbols", 2015, 11},
      {"csIBMSymbols", 2015, 12},
      {"IBM-Thai", 2016, 8},
      {"csIBMThai", 2016, 9},
      {"HP-Legal", 2017, 8},
      {"csHPLegal", 2017, 9},
      {"HP-Pi-font", 2018, 10},
      {"csHPPiFont", 2018, 10},
      {"HP-Math8", 2019, 8},
      {"csHPMath8", 2019, 9},
      {"Adobe-Symbol-Encoding", 2020, 21},
      {"csHPPSMath", 2020, 10},
      {"HP-DeskTop", 2021, 10},
      {"csHPDesktop", 2021, 11},
      {"Ventura-Math", 2022, 12},
      {"csVenturaMath", 2022, 13},
      {"Microsoft-Publishing", 2023, 20},
      {"csMicrosoftPublishing", 2023, 21},
      {"Windows-31J", 2024, 11},
      {"csWindows31J", 2024, 12},
      {"GB2312", 2025, 6},
      {"csGB2312", 2025, 8},
      {"Big5", 2026, 4},
      {"csBig5", 2026, 6},
      {"macintosh", 2027, 9},
      {"csMacintosh", 2027, 11},
      {"mac", 2027, 3},
      {"IBM037", 2028, 6},
      {"cp037", 2028, 5},
      {"csIBM037", 2028, 8},
      {"ebcdic-cp-ca", 2028, 12},
      {"ebcdic-cp-nl", 2028, 12},
      {"ebcdic-cp-us", 2028, 12},
      {"ebcdic-cp-wt", 2028, 12},
      {"IBM038", 2029, 6},
      {"EBCDIC-INT", 2029, 10},
      {"cp038", 2029, 5},
      {"csIBM038", 2029, 8},
      {"IBM273", 2030, 6},
      {"CP273", 2030, 5},
      {"csIBM273", 2030, 8},
      {"IBM274", 2031, 6},
      {"CP274", 2031, 5},
      {"EBCDIC-BE", 2031, 9},
      {"csIBM274", 2031, 8},
      {"IBM275", 2032, 6},
      {"EBCDIC-BR", 2032, 9},
      {"cp275", 2032, 5},
      {"csIBM275", 2032, 8},
      {"IBM277", 2033, 6},
      {"EBCDIC-CP-DK", 2033, 12},
      {"EBCDIC-CP-NO", 2033, 12},
      {"csIBM277", 2033, 8},
      {"IBM278", 2034, 6},
      {"CP278", 2034, 5},
      {"csIBM278", 2034, 8},
      {"ebcdic-cp-fi", 2034, 12},
      {"ebcdic-cp-se", 2034, 12},
      {"IBM280", 2035, 6},
      {"CP280", 2035, 5},
      {"csIBM280", 2035, 8},
      {"ebcdic-cp-it", 2035, 12},
      {"IBM281", 2036, 6},
      {"EBCDIC-JP-E", 2036, 11},
      {"cp281", 2036, 5},
      {"csIBM281", 2036, 8},
      {"IBM284", 2037, 6},
      {"CP284", 2037, 5},
      {"csIBM284", 2037, 8},
      {"ebcdic-cp-es", 2037, 12},
      {"IBM285", 2038, 6},
      {"CP285", 2038, 5},
      {"csIBM285", 2038, 8},
      {"ebcdic-cp-gb", 2038, 12},
      {"IBM290", 2039, 6},
      {"EBCDIC-JP-kana", 2039, 14},
      {"cp290", 2039, 5},
      {"csIBM290", 2039, 8},
      {"IBM297", 2040, 6},
      {"cp297", 2040, 5},
      {"csIBM297", 2040, 8},
      {"ebcdic-cp-fr", 2040, 12},
      {"IBM420", 2041, 6},
      {"cp420", 2041, 5},
      {"csIBM420", 2041, 8},
      {"ebcdic-cp-ar1", 2041, 13},
      {"IBM423", 2042, 6},
      {"cp423", 2042, 5},
      {"csIBM423", 2042, 8},
      {"ebcdic-cp-gr", 2042, 12},
      {"IBM424", 2043, 6},
      {"cp424", 2043, 5},
      {"csIBM424", 2043, 8},
      {"ebcdic-cp-he", 2043, 12},
      {"IBM500", 2044, 6},
      {"CP500", 2044, 5},
      {"csIBM500", 2044, 8},
      {"ebcdic-cp-be", 2044, 12},
      {"ebcdic-cp-ch", 2044, 12},
      {"IBM851", 2045, 6},
      {"851", 2045, 3},
      {"cp851", 2045, 5},
      {"csIBM851", 2045, 8},
      {"IBM855", 2046, 6},
      {"855", 2046, 3},
      {"cp855", 2046, 5},
      {"csIBM855", 2046, 8},
      {"IBM857", 2047, 6},
      {"857", 2047, 3},
      {"cp857", 2047, 5},
      {"csIBM857", 2047, 8},
      {"IBM860", 2048, 6},
      {"860", 2048, 3},
      {"cp860", 2048, 5},
      {"csIBM860", 2048, 8},
      {"IBM861", 2049, 6},
      {"861", 2049, 3},
      {"cp-is", 2049, 5},
      {"cp861", 2049, 5},
      {"csIBM861", 2049, 8},
      {"IBM863", 2050, 6},
      {"863", 2050, 3},
      {"cp863", 2050, 5},
      {"csIBM863", 2050, 8},
      {"IBM864", 2051, 6},
      {"cp864", 2051, 5},
      {"csIBM864", 2051, 8},
      {"IBM865", 2052, 6},
      {"865", 2052, 3},
      {"cp865", 2052, 5},
      {"csIBM865", 2052, 8},
      {"IBM868", 2053, 6},
      {"CP868", 2053, 5},
      {"cp-ar", 2053, 5},
      {"csIBM868", 2053, 8},
      {"IBM869", 2054, 6},
      {"869", 2054, 3},
      {"cp-gr", 2054, 5},
      {"cp869", 2054, 5},
      {"csIBM869", 2054, 8},
      {"IBM870", 2055, 6},
      {"CP870", 2055, 5},
      {"csIBM870", 2055, 8},
      {"ebcdic-cp-roece", 2055, 15},
      {"ebcdic-cp-yu", 2055, 12},
      {"IBM871", 2056, 6},
      {"CP871", 2056, 5},
      {"csIBM871", 2056, 8},
      {"ebcdic-cp-is", 2056, 12},
      {"IBM880", 2057, 6},
      {"EBCDIC-Cyrillic", 2057, 15},
      {"cp880", 2057, 5},
      {"csIBM880", 2057, 8},
      {"IBM891", 2058, 6},
      {"cp891", 2058, 5},
      {"csIBM891", 2058, 8},
      {"IBM903", 2059, 6},
      {"cp903", 2059, 5},
      {"csIBM903", 2059, 8},
      {"IBM904", 2060, 6},
      {"904", 2060, 3},
      {"cp904", 2060, 5},
      {"csIBBM904", 2060, 9},
      {"IBM905", 2061, 6},
      {"CP905", 2061, 5},
      {"csIBM905", 2061, 8},
      {"ebcdic-cp-tr", 2061, 12},
      {"IBM918", 2062, 6},
      {"CP918", 2062, 5},
      {"csIBM918", 2062, 8},
      {"ebcdic-cp-ar2", 2062, 13},
      {"IBM1026", 2063, 7},
      {"CP1026", 2063, 6},
      {"csIBM1026", 2063, 9},
      {"EBCDIC-AT-DE", 2064, 12},
      {"csIBMEBCDICATDE", 2064, 15},
      {"EBCDIC-AT-DE-A", 2065, 14},
      {"csEBCDICATDEA", 2065, 13},
      {"EBCDIC-CA-FR", 2066, 12},
      {"csEBCDICCAFR", 2066, 12},
      {"EBCDIC-DK-NO", 2067, 12},
      {"csEBCDICDKNO", 2067, 12},
      {"EBCDIC-DK-NO-A", 2068, 14},
      {"csEBCDICDKNOA", 2068, 13},
      {"EBCDIC-FI-SE", 2069, 12},
      {"csEBCDICFISE", 2069, 12},
      {"EBCDIC-FI-SE-A", 2070, 14},
      {"csEBCDICFISEA", 2070, 13},
      {"EBCDIC-FR", 2071, 9},
      {"csEBCDICFR", 2071, 10},
      {"EBCDIC-IT", 2072, 9},
      {"csEBCDICIT", 2072, 10},
      {"EBCDIC-PT", 2073, 9},
      {"csEBCDICPT", 2073, 10},
      {"EBCDIC-ES", 2074, 9},
      {"csEBCDICES", 2074, 10},
      {"EBCDIC-ES-A", 2075, 11},
      {"csEBCDICESA", 2075, 11},
      {"EBCDIC-ES-S", 2076, 11},
      {"csEBCDICESS", 2076, 11},
      {"EBCDIC-UK", 2077, 9},
      {"csEBCDICUK", 2077, 10},
      {"EBCDIC-US", 2078, 9},
      {"csEBCDICUS", 2078, 10},
      {"UNKNOWN-8BIT", 2079, 12},
      {"csUnknown8BiT", 2079, 13},
      {"MNEMONIC", 2080, 8},
      {"csMnemonic", 2080, 10},
      {"MNEM", 2081, 4},
      {"csMnem", 2081, 6},
      {"VISCII", 2082, 6},
      {"csVISCII", 2082, 8},
      {"VIQR", 2083, 4},
      {"csVIQR", 2083, 6},
      {"KOI8-R", 2084, 6},
      {"csKOI8R", 2084, 7},
      {"HZ-GB-2312", 2085, 10},
      {"IBM866", 2086, 6},
      {"866", 2086, 3},
      {"cp866", 2086, 5},
      {"csIBM866", 2086, 8},
      {"IBM775", 2087, 6},
      {"cp775", 2087, 5},
      {"csPC775Baltic", 2087, 13},
      {"KOI8-U", 2088, 6},
      {"csKOI8U", 2088, 7},
      {"IBM00858", 2089, 8},
      {"CCSID00858", 2089, 10},
      {"CP00858", 2089, 7},
      {"PC-Multilingual-850+euro", 2089, 24},
      {"csIBM00858", 2089, 10},
      {"IBM00924", 2090, 8},
      {"CCSID00924", 2090, 10},
      {"CP00924", 2090, 7},
      {"csIBM00924", 2090, 10},
      {"ebcdic-Latin9--euro", 2090, 19},
      {"IBM01140", 2091, 8},
      {"CCSID01140", 2091, 10},
      {"CP01140", 2091, 7},
      {"csIBM01140", 2091, 10},
      {"ebcdic-us-37+euro", 2091, 17},
      {"IBM01141", 2092, 8},
      {"CCSID01141", 2092, 10},
      {"CP01141", 2092, 7},
      {"csIBM01141", 2092, 10},
      {"ebcdic-de-273+euro", 2092, 18},
      {"IBM01142", 2093, 8},
      {"CCSID01142", 2093, 10},
      {"CP01142", 2093, 7},
      {"csIBM01142", 2093, 10},
      {"ebcdic-dk-277+euro", 2093, 18},
      {"ebcdic-no-277+euro", 2093, 18},
      {"IBM01143", 2094, 8},
      {"CCSID01143", 2094, 10},
      {"CP01143", 2094, 7},
      {"csIBM01143", 2094, 10},
      {"ebcdic-fi-278+euro", 2094, 18},
      {"ebcdic-se-278+euro", 2094, 18},
      {"IBM01144", 2095, 8},
      {"CCSID01144", 2095, 10},
      {"CP01144", 2095, 7},
      {"csIBM01144", 2095, 10},
      {"ebcdic-it-280+euro", 2095, 18},
      {"IBM01145", 2096, 8},
      {"CCSID01145", 2096, 10},
      {"CP01145", 2096, 7},
      {"csIBM01145", 2096, 10},
      {"ebcdic-es-284+euro", 2096, 18},
      {"IBM01146", 2097, 8},
      {"CCSID01146", 2097, 10},
      {"CP01146", 2097, 7},
      {"csIBM01146", 2097, 10},
      {"ebcdic-gb-285+euro", 2097, 18},
      {"IBM01147", 2098, 8},
      {"CCSID01147", 2098, 10},
      {"CP01147", 2098, 7},
      {"csIBM01147", 2098, 10},
      {"ebcdic-fr-297+euro", 2098, 18},
      {"IBM01148", 2099, 8},
      {"CCSID01148", 2099, 10},
      {"CP01148", 2099, 7},
      {"csIBM01148", 2099, 10},
      {"ebcdic-international-500+euro", 2099, 29},
      {"IBM01149", 2100, 8},
      {"CCSID01149", 2100, 10},
      {"CP01149", 2100, 7},
      {"csIBM01149", 2100, 10},
      {"ebcdic-is-871+euro", 2100, 18},
      {"Big5-HKSCS", 2101, 10},
      {"csBig5HKSCS", 2101, 11},
      {"IBM1047", 2102, 7},
      {"IBM-1047", 2102, 8},
      {"csIBM1047", 2102, 9},
      {"PTCP154", 2103, 7},
      {"CP154", 2103, 5},
      {"Cyrillic-Asian", 2103, 14},
      {"PT154", 2103, 5},
      {"csPTCP154", 2103, 9},
      {"Amiga-1251", 2104, 10},
      {"Ami-1251", 2104, 8},
      {"Ami1251", 2104, 7},
      {"Amiga1251", 2104, 9},
      {"csAmiga1251", 2104, 11},
      {"KOI7-switched", 2105, 13},
      {"csKOI7switched", 2105, 14},
      {"BRF", 2106, 3},
      {"csBRF", 2106, 5},
      {"TSCII", 2107, 5},
      {"csTSCII", 2107, 7},
      {"CP51932", 2108, 7},
      {"csCP51932", 2108, 9},
      {"windows-874", 2109, 11},
      {"cswindows874", 2109, 12},
      {"windows-1250", 2250, 12},
      {"cswindows1250", 2250, 13},
      {"windows-1251", 2251, 12},
      {"cswindows1251", 2251, 13},
      {"windows-1252", 2252, 12},
      {"cswindows1252", 2252, 13},
      {"windows-1253", 2253, 12},
      {"cswindows1253", 2253, 13},
      {"windows-1254", 2254, 12},
      {"cswindows1254", 2254, 13},
      {"windows-1255", 2255, 12},
      {"cswindows1255", 2255, 13},
      {"windows-1256", 2256, 12},
      {"cswindows1256", 2256, 13},
      {"windows-1257", 2257, 12},
      {"cswindows1257", 2257, 13},
      {"windows-1258", 2258, 12},
      {"cswindows1258", 2258, 13},
      {"TIS-620", 2259, 7},
      {"ISO-8859-11", 2259, 11},
      {"csTIS620", 2259, 8},
      {"CP50220", 2260, 7},
      {"csCP50220", 2260, 9},
  };

  const __encoding_data* __encoding_rep_ = __text_encoding_data + 1;
  char __name_[max_name_length + 1]      = {0};
};

template <>
struct hash<text_encoding> {
  size_t operator()(const text_encoding& __enc) const noexcept { return std::hash<text_encoding::id>()(__enc.mib()); }
};

template <>
inline constexpr bool ranges::enable_borrowed_range<text_encoding::aliases_view> = true;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H
