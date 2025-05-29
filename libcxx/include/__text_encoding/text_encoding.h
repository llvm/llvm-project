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
    __id_rep __mib_rep_;
    const char* __name_;
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
    auto __rep = __encoding_rep_;
    if (__encoding_rep_->__name_[0]) {
      while (__rep > std::begin(__text_encoding_data) && __rep[-1].__mib_rep_ == __encoding_rep_->__mib_rep_) {
        __rep--;
      }
    } else {
      __rep = nullptr;
    }

    return aliases_view(__rep);
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
  };

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

    return __found_data != __data_last ? __found_data : __text_encoding_data; // other
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
      {1, "", 0},
      {2, "", 0},
      {3, "ANSI_X3.4-1968", 14},
      {3, "ANSI_X3.4-1986", 14},
      {3, "IBM367", 6},
      {3, "ISO646-US", 9},
      {3, "ISO_646.irv:1991", 16},
      {3, "cp367", 5},
      {3, "csASCII", 7},
      {3, "iso-ir-6", 8},
      {3, "us", 2},
      {4, "ISO-8859-1", 10},
      {4, "ISO_8859-1:1987", 15},
      {4, "CP819", 5},
      {4, "IBM819", 6},
      {4, "ISO_8859-1", 10},
      {4, "csISOLatin1", 11},
      {4, "iso-ir-100", 10},
      {4, "l1", 2},
      {4, "latin1", 6},
      {5, "ISO-8859-2", 10},
      {5, "ISO_8859-2:1987", 15},
      {5, "ISO_8859-2", 10},
      {5, "csISOLatin2", 11},
      {5, "iso-ir-101", 10},
      {5, "l2", 2},
      {5, "latin2", 6},
      {6, "ISO-8859-3", 10},
      {6, "ISO_8859-3:1988", 15},
      {6, "ISO_8859-3", 10},
      {6, "csISOLatin3", 11},
      {6, "iso-ir-109", 10},
      {6, "l3", 2},
      {6, "latin3", 6},
      {7, "ISO-8859-4", 10},
      {7, "ISO_8859-4:1988", 15},
      {7, "ISO_8859-4", 10},
      {7, "csISOLatin4", 11},
      {7, "iso-ir-110", 10},
      {7, "l4", 2},
      {7, "latin4", 6},
      {8, "ISO-8859-5", 10},
      {8, "ISO_8859-5:1988", 15},
      {8, "ISO_8859-5", 10},
      {8, "csISOLatinCyrillic", 18},
      {8, "cyrillic", 8},
      {8, "iso-ir-144", 10},
      {9, "ISO-8859-6", 10},
      {9, "ISO_8859-6:1987", 15},
      {9, "ASMO-708", 8},
      {9, "ECMA-114", 8},
      {9, "ISO_8859-6", 10},
      {9, "arabic", 6},
      {9, "csISOLatinArabic", 16},
      {9, "iso-ir-127", 10},
      {10, "ISO-8859-7", 10},
      {10, "ISO_8859-7:1987", 15},
      {10, "ECMA-118", 8},
      {10, "ELOT_928", 8},
      {10, "ISO_8859-7", 10},
      {10, "csISOLatinGreek", 15},
      {10, "greek", 5},
      {10, "greek8", 6},
      {10, "iso-ir-126", 10},
      {11, "ISO-8859-8", 10},
      {11, "ISO_8859-8:1988", 15},
      {11, "ISO_8859-8", 10},
      {11, "csISOLatinHebrew", 16},
      {11, "hebrew", 6},
      {11, "iso-ir-138", 10},
      {12, "ISO-8859-9", 10},
      {12, "ISO_8859-9:1989", 15},
      {12, "ISO_8859-9", 10},
      {12, "csISOLatin5", 11},
      {12, "iso-ir-148", 10},
      {12, "l5", 2},
      {12, "latin5", 6},
      {13, "ISO-8859-10", 11},
      {13, "ISO_8859-10:1992", 16},
      {13, "csISOLatin6", 11},
      {13, "iso-ir-157", 10},
      {13, "l6", 2},
      {13, "latin6", 6},
      {14, "ISO_6937-2-add", 14},
      {14, "csISOTextComm", 13},
      {14, "iso-ir-142", 10},
      {15, "JIS_X0201", 9},
      {15, "X0201", 5},
      {15, "csHalfWidthKatakana", 19},
      {16, "JIS_Encoding", 12},
      {16, "csJISEncoding", 13},
      {17, "Shift_JIS", 9},
      {17, "MS_Kanji", 8},
      {17, "csShiftJIS", 10},
      {18, "EUC-JP", 6},
      {18, "Extended_UNIX_Code_Packed_Format_for_Japanese", 45},
      {18, "csEUCPkdFmtJapanese", 19},
      {19, "Extended_UNIX_Code_Fixed_Width_for_Japanese", 43},
      {19, "csEUCFixWidJapanese", 19},
      {20, "BS_4730", 7},
      {20, "ISO646-GB", 9},
      {20, "csISO4UnitedKingdom", 19},
      {20, "gb", 2},
      {20, "iso-ir-4", 8},
      {20, "uk", 2},
      {21, "SEN_850200_C", 12},
      {21, "ISO646-SE2", 10},
      {21, "csISO11SwedishForNames", 22},
      {21, "iso-ir-11", 9},
      {21, "se2", 3},
      {22, "IT", 2},
      {22, "ISO646-IT", 9},
      {22, "csISO15Italian", 14},
      {22, "iso-ir-15", 9},
      {23, "ES", 2},
      {23, "ISO646-ES", 9},
      {23, "csISO17Spanish", 14},
      {23, "iso-ir-17", 9},
      {24, "DIN_66003", 9},
      {24, "ISO646-DE", 9},
      {24, "csISO21German", 13},
      {24, "de", 2},
      {24, "iso-ir-21", 9},
      {25, "NS_4551-1", 9},
      {25, "ISO646-NO", 9},
      {25, "csISO60DanishNorwegian", 22},
      {25, "csISO60Norwegian1", 17},
      {25, "iso-ir-60", 9},
      {25, "no", 2},
      {26, "NF_Z_62-010", 11},
      {26, "ISO646-FR", 9},
      {26, "csISO69French", 13},
      {26, "fr", 2},
      {26, "iso-ir-69", 9},
      {27, "ISO-10646-UTF-1", 15},
      {27, "csISO10646UTF1", 14},
      {28, "ISO_646.basic:1983", 18},
      {28, "csISO646basic1983", 17},
      {28, "ref", 3},
      {29, "INVARIANT", 9},
      {29, "csINVARIANT", 11},
      {30, "ISO_646.irv:1983", 16},
      {30, "csISO2IntlRefVersion", 20},
      {30, "irv", 3},
      {30, "iso-ir-2", 8},
      {31, "NATS-SEFI", 9},
      {31, "csNATSSEFI", 10},
      {31, "iso-ir-8-1", 10},
      {32, "NATS-SEFI-ADD", 13},
      {32, "csNATSSEFIADD", 13},
      {32, "iso-ir-8-2", 10},
      {35, "SEN_850200_B", 12},
      {35, "FI", 2},
      {35, "ISO646-FI", 9},
      {35, "ISO646-SE", 9},
      {35, "csISO10Swedish", 14},
      {35, "iso-ir-10", 9},
      {35, "se", 2},
      {36, "KS_C_5601-1987", 14},
      {36, "KSC_5601", 8},
      {36, "KS_C_5601-1989", 14},
      {36, "csKSC56011987", 13},
      {36, "iso-ir-149", 10},
      {36, "korean", 6},
      {37, "ISO-2022-KR", 11},
      {37, "csISO2022KR", 11},
      {38, "EUC-KR", 6},
      {38, "csEUCKR", 7},
      {39, "ISO-2022-JP", 11},
      {39, "csISO2022JP", 11},
      {40, "ISO-2022-JP-2", 13},
      {40, "csISO2022JP2", 12},
      {41, "JIS_C6220-1969-jp", 17},
      {41, "JIS_C6220-1969", 14},
      {41, "csISO13JISC6220jp", 17},
      {41, "iso-ir-13", 9},
      {41, "katakana", 8},
      {41, "x0201-7", 7},
      {42, "JIS_C6220-1969-ro", 17},
      {42, "ISO646-JP", 9},
      {42, "csISO14JISC6220ro", 17},
      {42, "iso-ir-14", 9},
      {42, "jp", 2},
      {43, "PT", 2},
      {43, "ISO646-PT", 9},
      {43, "csISO16Portuguese", 17},
      {43, "iso-ir-16", 9},
      {44, "greek7-old", 10},
      {44, "csISO18Greek7Old", 16},
      {44, "iso-ir-18", 9},
      {45, "latin-greek", 11},
      {45, "csISO19LatinGreek", 17},
      {45, "iso-ir-19", 9},
      {46, "NF_Z_62-010_(1973)", 18},
      {46, "ISO646-FR1", 10},
      {46, "csISO25French", 13},
      {46, "iso-ir-25", 9},
      {47, "Latin-greek-1", 13},
      {47, "csISO27LatinGreek1", 18},
      {47, "iso-ir-27", 9},
      {48, "ISO_5427", 8},
      {48, "csISO5427Cyrillic", 17},
      {48, "iso-ir-37", 9},
      {49, "JIS_C6226-1978", 14},
      {49, "csISO42JISC62261978", 19},
      {49, "iso-ir-42", 9},
      {50, "BS_viewdata", 11},
      {50, "csISO47BSViewdata", 17},
      {50, "iso-ir-47", 9},
      {51, "INIS", 4},
      {51, "csISO49INIS", 11},
      {51, "iso-ir-49", 9},
      {52, "INIS-8", 6},
      {52, "csISO50INIS8", 12},
      {52, "iso-ir-50", 9},
      {53, "INIS-cyrillic", 13},
      {53, "csISO51INISCyrillic", 19},
      {53, "iso-ir-51", 9},
      {54, "ISO_5427:1981", 13},
      {54, "ISO5427Cyrillic1981", 19},
      {54, "csISO54271981", 13},
      {54, "iso-ir-54", 9},
      {55, "ISO_5428:1980", 13},
      {55, "csISO5428Greek", 14},
      {55, "iso-ir-55", 9},
      {56, "GB_1988-80", 10},
      {56, "ISO646-CN", 9},
      {56, "cn", 2},
      {56, "csISO57GB1988", 13},
      {56, "iso-ir-57", 9},
      {57, "GB_2312-80", 10},
      {57, "chinese", 7},
      {57, "csISO58GB231280", 15},
      {57, "iso-ir-58", 9},
      {58, "NS_4551-2", 9},
      {58, "ISO646-NO2", 10},
      {58, "csISO61Norwegian2", 17},
      {58, "iso-ir-61", 9},
      {58, "no2", 3},
      {59, "videotex-suppl", 14},
      {59, "csISO70VideotexSupp1", 20},
      {59, "iso-ir-70", 9},
      {60, "PT2", 3},
      {60, "ISO646-PT2", 10},
      {60, "csISO84Portuguese2", 18},
      {60, "iso-ir-84", 9},
      {61, "ES2", 3},
      {61, "ISO646-ES2", 10},
      {61, "csISO85Spanish2", 15},
      {61, "iso-ir-85", 9},
      {62, "MSZ_7795.3", 10},
      {62, "ISO646-HU", 9},
      {62, "csISO86Hungarian", 16},
      {62, "hu", 2},
      {62, "iso-ir-86", 9},
      {63, "JIS_C6226-1983", 14},
      {63, "JIS_X0208-1983", 14},
      {63, "csISO87JISX0208", 15},
      {63, "iso-ir-87", 9},
      {63, "x0208", 5},
      {64, "greek7", 6},
      {64, "csISO88Greek7", 13},
      {64, "iso-ir-88", 9},
      {65, "ASMO_449", 8},
      {65, "ISO_9036", 8},
      {65, "arabic7", 7},
      {65, "csISO89ASMO449", 14},
      {65, "iso-ir-89", 9},
      {66, "iso-ir-90", 9},
      {66, "csISO90", 7},
      {67, "JIS_C6229-1984-a", 16},
      {67, "csISO91JISC62291984a", 20},
      {67, "iso-ir-91", 9},
      {67, "jp-ocr-a", 8},
      {68, "JIS_C6229-1984-b", 16},
      {68, "ISO646-JP-OCR-B", 15},
      {68, "csISO92JISC62991984b", 20},
      {68, "iso-ir-92", 9},
      {68, "jp-ocr-b", 8},
      {69, "JIS_C6229-1984-b-add", 20},
      {69, "csISO93JIS62291984badd", 22},
      {69, "iso-ir-93", 9},
      {69, "jp-ocr-b-add", 12},
      {70, "JIS_C6229-1984-hand", 19},
      {70, "csISO94JIS62291984hand", 22},
      {70, "iso-ir-94", 9},
      {70, "jp-ocr-hand", 11},
      {71, "JIS_C6229-1984-hand-add", 23},
      {71, "csISO95JIS62291984handadd", 25},
      {71, "iso-ir-95", 9},
      {71, "jp-ocr-hand-add", 15},
      {72, "JIS_C6229-1984-kana", 19},
      {72, "csISO96JISC62291984kana", 23},
      {72, "iso-ir-96", 9},
      {73, "ISO_2033-1983", 13},
      {73, "csISO2033", 9},
      {73, "e13b", 4},
      {73, "iso-ir-98", 9},
      {74, "ANSI_X3.110-1983", 16},
      {74, "CSA_T500-1983", 13},
      {74, "NAPLPS", 6},
      {74, "csISO99NAPLPS", 13},
      {74, "iso-ir-99", 9},
      {75, "T.61-7bit", 9},
      {75, "csISO102T617bit", 15},
      {75, "iso-ir-102", 10},
      {76, "T.61-8bit", 9},
      {76, "T.61", 4},
      {76, "csISO103T618bit", 15},
      {76, "iso-ir-103", 10},
      {77, "ECMA-cyrillic", 13},
      {77, "KOI8-E", 6},
      {77, "csISO111ECMACyrillic", 20},
      {77, "iso-ir-111", 10},
      {78, "CSA_Z243.4-1985-1", 17},
      {78, "ISO646-CA", 9},
      {78, "ca", 2},
      {78, "csISO121Canadian1", 17},
      {78, "csa7-1", 6},
      {78, "csa71", 5},
      {78, "iso-ir-121", 10},
      {79, "CSA_Z243.4-1985-2", 17},
      {79, "ISO646-CA2", 10},
      {79, "csISO122Canadian2", 17},
      {79, "csa7-2", 6},
      {79, "csa72", 5},
      {79, "iso-ir-122", 10},
      {80, "CSA_Z243.4-1985-gr", 18},
      {80, "csISO123CSAZ24341985gr", 22},
      {80, "iso-ir-123", 10},
      {81, "ISO-8859-6-E", 12},
      {81, "ISO_8859-6-E", 12},
      {81, "csISO88596E", 11},
      {82, "ISO-8859-6-I", 12},
      {82, "ISO_8859-6-I", 12},
      {82, "csISO88596I", 11},
      {83, "T.101-G2", 8},
      {83, "csISO128T101G2", 14},
      {83, "iso-ir-128", 10},
      {84, "ISO-8859-8-E", 12},
      {84, "ISO_8859-8-E", 12},
      {84, "csISO88598E", 11},
      {85, "ISO-8859-8-I", 12},
      {85, "ISO_8859-8-I", 12},
      {85, "csISO88598I", 11},
      {86, "CSN_369103", 10},
      {86, "csISO139CSN369103", 17},
      {86, "iso-ir-139", 10},
      {87, "JUS_I.B1.002", 12},
      {87, "ISO646-YU", 9},
      {87, "csISO141JUSIB1002", 17},
      {87, "iso-ir-141", 10},
      {87, "js", 2},
      {87, "yu", 2},
      {88, "IEC_P27-1", 9},
      {88, "csISO143IECP271", 15},
      {88, "iso-ir-143", 10},
      {89, "JUS_I.B1.003-serb", 17},
      {89, "csISO146Serbian", 15},
      {89, "iso-ir-146", 10},
      {89, "serbian", 7},
      {90, "JUS_I.B1.003-mac", 16},
      {90, "csISO147Macedonian", 18},
      {90, "iso-ir-147", 10},
      {90, "macedonian", 10},
      {91, "greek-ccitt", 11},
      {91, "csISO150", 8},
      {91, "csISO150GreekCCITT", 18},
      {91, "iso-ir-150", 10},
      {92, "NC_NC00-10:81", 13},
      {92, "ISO646-CU", 9},
      {92, "csISO151Cuba", 12},
      {92, "cuba", 4},
      {92, "iso-ir-151", 10},
      {93, "ISO_6937-2-25", 13},
      {93, "csISO6937Add", 12},
      {93, "iso-ir-152", 10},
      {94, "GOST_19768-74", 13},
      {94, "ST_SEV_358-88", 13},
      {94, "csISO153GOST1976874", 19},
      {94, "iso-ir-153", 10},
      {95, "ISO_8859-supp", 13},
      {95, "csISO8859Supp", 13},
      {95, "iso-ir-154", 10},
      {95, "latin1-2-5", 10},
      {96, "ISO_10367-box", 13},
      {96, "csISO10367Box", 13},
      {96, "iso-ir-155", 10},
      {97, "latin-lap", 9},
      {97, "csISO158Lap", 11},
      {97, "iso-ir-158", 10},
      {97, "lap", 3},
      {98, "JIS_X0212-1990", 14},
      {98, "csISO159JISX02121990", 20},
      {98, "iso-ir-159", 10},
      {98, "x0212", 5},
      {99, "DS_2089", 7},
      {99, "DS2089", 6},
      {99, "ISO646-DK", 9},
      {99, "csISO646Danish", 14},
      {99, "dk", 2},
      {100, "us-dk", 5},
      {100, "csUSDK", 6},
      {101, "dk-us", 5},
      {101, "csDKUS", 6},
      {102, "KSC5636", 7},
      {102, "ISO646-KR", 9},
      {102, "csKSC5636", 9},
      {103, "UNICODE-1-1-UTF-7", 17},
      {103, "csUnicode11UTF7", 15},
      {104, "ISO-2022-CN", 11},
      {104, "csISO2022CN", 11},
      {105, "ISO-2022-CN-EXT", 15},
      {105, "csISO2022CNEXT", 14},
      {106, "UTF-8", 5},
      {106, "csUTF8", 6},
      {109, "ISO-8859-13", 11},
      {109, "csISO885913", 11},
      {110, "ISO-8859-14", 11},
      {110, "ISO_8859-14", 11},
      {110, "ISO_8859-14:1998", 16},
      {110, "csISO885914", 11},
      {110, "iso-celtic", 10},
      {110, "iso-ir-199", 10},
      {110, "l8", 2},
      {110, "latin8", 6},
      {111, "ISO-8859-15", 11},
      {111, "ISO_8859-15", 11},
      {111, "Latin-9", 7},
      {111, "csISO885915", 11},
      {112, "ISO-8859-16", 11},
      {112, "ISO_8859-16", 11},
      {112, "ISO_8859-16:2001", 16},
      {112, "csISO885916", 11},
      {112, "iso-ir-226", 10},
      {112, "l10", 3},
      {112, "latin10", 7},
      {113, "GBK", 3},
      {113, "CP936", 5},
      {113, "MS936", 5},
      {113, "csGBK", 5},
      {113, "windows-936", 11},
      {114, "GB18030", 7},
      {114, "csGB18030", 9},
      {115, "OSD_EBCDIC_DF04_15", 18},
      {115, "csOSDEBCDICDF0415", 17},
      {116, "OSD_EBCDIC_DF03_IRV", 19},
      {116, "csOSDEBCDICDF03IRV", 18},
      {117, "OSD_EBCDIC_DF04_1", 17},
      {117, "csOSDEBCDICDF041", 16},
      {118, "ISO-11548-1", 11},
      {118, "ISO_11548-1", 11},
      {118, "ISO_TR_11548-1", 14},
      {118, "csISO115481", 11},
      {119, "KZ-1048", 7},
      {119, "RK1048", 6},
      {119, "STRK1048-2002", 13},
      {119, "csKZ1048", 8},
      {1000, "ISO-10646-UCS-2", 15},
      {1000, "csUnicode", 9},
      {1001, "ISO-10646-UCS-4", 15},
      {1001, "csUCS4", 6},
      {1002, "ISO-10646-UCS-Basic", 19},
      {1002, "csUnicodeASCII", 14},
      {1003, "ISO-10646-Unicode-Latin1", 24},
      {1003, "ISO-10646", 9},
      {1003, "csUnicodeLatin1", 15},
      {1004, "ISO-10646-J-1", 13},
      {1004, "csUnicodeJapanese", 17},
      {1005, "ISO-Unicode-IBM-1261", 20},
      {1005, "csUnicodeIBM1261", 16},
      {1006, "ISO-Unicode-IBM-1268", 20},
      {1006, "csUnicodeIBM1268", 16},
      {1007, "ISO-Unicode-IBM-1276", 20},
      {1007, "csUnicodeIBM1276", 16},
      {1008, "ISO-Unicode-IBM-1264", 20},
      {1008, "csUnicodeIBM1264", 16},
      {1009, "ISO-Unicode-IBM-1265", 20},
      {1009, "csUnicodeIBM1265", 16},
      {1010, "UNICODE-1-1", 11},
      {1010, "csUnicode11", 11},
      {1011, "SCSU", 4},
      {1011, "csSCSU", 6},
      {1012, "UTF-7", 5},
      {1012, "csUTF7", 6},
      {1013, "UTF-16BE", 8},
      {1013, "csUTF16BE", 9},
      {1014, "UTF-16LE", 8},
      {1014, "csUTF16LE", 9},
      {1015, "UTF-16", 6},
      {1015, "csUTF16", 7},
      {1016, "CESU-8", 6},
      {1016, "csCESU-8", 8},
      {1016, "csCESU8", 7},
      {1017, "UTF-32", 6},
      {1017, "csUTF32", 7},
      {1018, "UTF-32BE", 8},
      {1018, "csUTF32BE", 9},
      {1019, "UTF-32LE", 8},
      {1019, "csUTF32LE", 9},
      {1020, "BOCU-1", 6},
      {1020, "csBOCU-1", 8},
      {1020, "csBOCU1", 7},
      {1021, "UTF-7-IMAP", 10},
      {1021, "csUTF7IMAP", 10},
      {2000, "ISO-8859-1-Windows-3.0-Latin-1", 30},
      {2000, "csWindows30Latin1", 17},
      {2001, "ISO-8859-1-Windows-3.1-Latin-1", 30},
      {2001, "csWindows31Latin1", 17},
      {2002, "ISO-8859-2-Windows-Latin-2", 26},
      {2002, "csWindows31Latin2", 17},
      {2003, "ISO-8859-9-Windows-Latin-5", 26},
      {2003, "csWindows31Latin5", 17},
      {2004, "hp-roman8", 9},
      {2004, "csHPRoman8", 10},
      {2004, "r8", 2},
      {2004, "roman8", 6},
      {2005, "Adobe-Standard-Encoding", 23},
      {2005, "csAdobeStandardEncoding", 23},
      {2006, "Ventura-US", 10},
      {2006, "csVenturaUS", 11},
      {2007, "Ventura-International", 21},
      {2007, "csVenturaInternational", 22},
      {2008, "DEC-MCS", 7},
      {2008, "csDECMCS", 8},
      {2008, "dec", 3},
      {2009, "IBM850", 6},
      {2009, "850", 3},
      {2009, "cp850", 5},
      {2009, "csPC850Multilingual", 19},
      {2010, "IBM852", 6},
      {2010, "852", 3},
      {2010, "cp852", 5},
      {2010, "csPCp852", 8},
      {2011, "IBM437", 6},
      {2011, "437", 3},
      {2011, "cp437", 5},
      {2011, "csPC8CodePage437", 16},
      {2012, "PC8-Danish-Norwegian", 20},
      {2012, "csPC8DanishNorwegian", 20},
      {2013, "IBM862", 6},
      {2013, "862", 3},
      {2013, "cp862", 5},
      {2013, "csPC862LatinHebrew", 18},
      {2014, "PC8-Turkish", 11},
      {2014, "csPC8Turkish", 12},
      {2015, "IBM-Symbols", 11},
      {2015, "csIBMSymbols", 12},
      {2016, "IBM-Thai", 8},
      {2016, "csIBMThai", 9},
      {2017, "HP-Legal", 8},
      {2017, "csHPLegal", 9},
      {2018, "HP-Pi-font", 10},
      {2018, "csHPPiFont", 10},
      {2019, "HP-Math8", 8},
      {2019, "csHPMath8", 9},
      {2020, "Adobe-Symbol-Encoding", 21},
      {2020, "csHPPSMath", 10},
      {2021, "HP-DeskTop", 10},
      {2021, "csHPDesktop", 11},
      {2022, "Ventura-Math", 12},
      {2022, "csVenturaMath", 13},
      {2023, "Microsoft-Publishing", 20},
      {2023, "csMicrosoftPublishing", 21},
      {2024, "Windows-31J", 11},
      {2024, "csWindows31J", 12},
      {2025, "GB2312", 6},
      {2025, "csGB2312", 8},
      {2026, "Big5", 4},
      {2026, "csBig5", 6},
      {2027, "macintosh", 9},
      {2027, "csMacintosh", 11},
      {2027, "mac", 3},
      {2028, "IBM037", 6},
      {2028, "cp037", 5},
      {2028, "csIBM037", 8},
      {2028, "ebcdic-cp-ca", 12},
      {2028, "ebcdic-cp-nl", 12},
      {2028, "ebcdic-cp-us", 12},
      {2028, "ebcdic-cp-wt", 12},
      {2029, "IBM038", 6},
      {2029, "EBCDIC-INT", 10},
      {2029, "cp038", 5},
      {2029, "csIBM038", 8},
      {2030, "IBM273", 6},
      {2030, "CP273", 5},
      {2030, "csIBM273", 8},
      {2031, "IBM274", 6},
      {2031, "CP274", 5},
      {2031, "EBCDIC-BE", 9},
      {2031, "csIBM274", 8},
      {2032, "IBM275", 6},
      {2032, "EBCDIC-BR", 9},
      {2032, "cp275", 5},
      {2032, "csIBM275", 8},
      {2033, "IBM277", 6},
      {2033, "EBCDIC-CP-DK", 12},
      {2033, "EBCDIC-CP-NO", 12},
      {2033, "csIBM277", 8},
      {2034, "IBM278", 6},
      {2034, "CP278", 5},
      {2034, "csIBM278", 8},
      {2034, "ebcdic-cp-fi", 12},
      {2034, "ebcdic-cp-se", 12},
      {2035, "IBM280", 6},
      {2035, "CP280", 5},
      {2035, "csIBM280", 8},
      {2035, "ebcdic-cp-it", 12},
      {2036, "IBM281", 6},
      {2036, "EBCDIC-JP-E", 11},
      {2036, "cp281", 5},
      {2036, "csIBM281", 8},
      {2037, "IBM284", 6},
      {2037, "CP284", 5},
      {2037, "csIBM284", 8},
      {2037, "ebcdic-cp-es", 12},
      {2038, "IBM285", 6},
      {2038, "CP285", 5},
      {2038, "csIBM285", 8},
      {2038, "ebcdic-cp-gb", 12},
      {2039, "IBM290", 6},
      {2039, "EBCDIC-JP-kana", 14},
      {2039, "cp290", 5},
      {2039, "csIBM290", 8},
      {2040, "IBM297", 6},
      {2040, "cp297", 5},
      {2040, "csIBM297", 8},
      {2040, "ebcdic-cp-fr", 12},
      {2041, "IBM420", 6},
      {2041, "cp420", 5},
      {2041, "csIBM420", 8},
      {2041, "ebcdic-cp-ar1", 13},
      {2042, "IBM423", 6},
      {2042, "cp423", 5},
      {2042, "csIBM423", 8},
      {2042, "ebcdic-cp-gr", 12},
      {2043, "IBM424", 6},
      {2043, "cp424", 5},
      {2043, "csIBM424", 8},
      {2043, "ebcdic-cp-he", 12},
      {2044, "IBM500", 6},
      {2044, "CP500", 5},
      {2044, "csIBM500", 8},
      {2044, "ebcdic-cp-be", 12},
      {2044, "ebcdic-cp-ch", 12},
      {2045, "IBM851", 6},
      {2045, "851", 3},
      {2045, "cp851", 5},
      {2045, "csIBM851", 8},
      {2046, "IBM855", 6},
      {2046, "855", 3},
      {2046, "cp855", 5},
      {2046, "csIBM855", 8},
      {2047, "IBM857", 6},
      {2047, "857", 3},
      {2047, "cp857", 5},
      {2047, "csIBM857", 8},
      {2048, "IBM860", 6},
      {2048, "860", 3},
      {2048, "cp860", 5},
      {2048, "csIBM860", 8},
      {2049, "IBM861", 6},
      {2049, "861", 3},
      {2049, "cp-is", 5},
      {2049, "cp861", 5},
      {2049, "csIBM861", 8},
      {2050, "IBM863", 6},
      {2050, "863", 3},
      {2050, "cp863", 5},
      {2050, "csIBM863", 8},
      {2051, "IBM864", 6},
      {2051, "cp864", 5},
      {2051, "csIBM864", 8},
      {2052, "IBM865", 6},
      {2052, "865", 3},
      {2052, "cp865", 5},
      {2052, "csIBM865", 8},
      {2053, "IBM868", 6},
      {2053, "CP868", 5},
      {2053, "cp-ar", 5},
      {2053, "csIBM868", 8},
      {2054, "IBM869", 6},
      {2054, "869", 3},
      {2054, "cp-gr", 5},
      {2054, "cp869", 5},
      {2054, "csIBM869", 8},
      {2055, "IBM870", 6},
      {2055, "CP870", 5},
      {2055, "csIBM870", 8},
      {2055, "ebcdic-cp-roece", 15},
      {2055, "ebcdic-cp-yu", 12},
      {2056, "IBM871", 6},
      {2056, "CP871", 5},
      {2056, "csIBM871", 8},
      {2056, "ebcdic-cp-is", 12},
      {2057, "IBM880", 6},
      {2057, "EBCDIC-Cyrillic", 15},
      {2057, "cp880", 5},
      {2057, "csIBM880", 8},
      {2058, "IBM891", 6},
      {2058, "cp891", 5},
      {2058, "csIBM891", 8},
      {2059, "IBM903", 6},
      {2059, "cp903", 5},
      {2059, "csIBM903", 8},
      {2060, "IBM904", 6},
      {2060, "904", 3},
      {2060, "cp904", 5},
      {2060, "csIBBM904", 9},
      {2061, "IBM905", 6},
      {2061, "CP905", 5},
      {2061, "csIBM905", 8},
      {2061, "ebcdic-cp-tr", 12},
      {2062, "IBM918", 6},
      {2062, "CP918", 5},
      {2062, "csIBM918", 8},
      {2062, "ebcdic-cp-ar2", 13},
      {2063, "IBM1026", 7},
      {2063, "CP1026", 6},
      {2063, "csIBM1026", 9},
      {2064, "EBCDIC-AT-DE", 12},
      {2064, "csIBMEBCDICATDE", 15},
      {2065, "EBCDIC-AT-DE-A", 14},
      {2065, "csEBCDICATDEA", 13},
      {2066, "EBCDIC-CA-FR", 12},
      {2066, "csEBCDICCAFR", 12},
      {2067, "EBCDIC-DK-NO", 12},
      {2067, "csEBCDICDKNO", 12},
      {2068, "EBCDIC-DK-NO-A", 14},
      {2068, "csEBCDICDKNOA", 13},
      {2069, "EBCDIC-FI-SE", 12},
      {2069, "csEBCDICFISE", 12},
      {2070, "EBCDIC-FI-SE-A", 14},
      {2070, "csEBCDICFISEA", 13},
      {2071, "EBCDIC-FR", 9},
      {2071, "csEBCDICFR", 10},
      {2072, "EBCDIC-IT", 9},
      {2072, "csEBCDICIT", 10},
      {2073, "EBCDIC-PT", 9},
      {2073, "csEBCDICPT", 10},
      {2074, "EBCDIC-ES", 9},
      {2074, "csEBCDICES", 10},
      {2075, "EBCDIC-ES-A", 11},
      {2075, "csEBCDICESA", 11},
      {2076, "EBCDIC-ES-S", 11},
      {2076, "csEBCDICESS", 11},
      {2077, "EBCDIC-UK", 9},
      {2077, "csEBCDICUK", 10},
      {2078, "EBCDIC-US", 9},
      {2078, "csEBCDICUS", 10},
      {2079, "UNKNOWN-8BIT", 12},
      {2079, "csUnknown8BiT", 13},
      {2080, "MNEMONIC", 8},
      {2080, "csMnemonic", 10},
      {2081, "MNEM", 4},
      {2081, "csMnem", 6},
      {2082, "VISCII", 6},
      {2082, "csVISCII", 8},
      {2083, "VIQR", 4},
      {2083, "csVIQR", 6},
      {2084, "KOI8-R", 6},
      {2084, "csKOI8R", 7},
      {2085, "HZ-GB-2312", 10},
      {2086, "IBM866", 6},
      {2086, "866", 3},
      {2086, "cp866", 5},
      {2086, "csIBM866", 8},
      {2087, "IBM775", 6},
      {2087, "cp775", 5},
      {2087, "csPC775Baltic", 13},
      {2088, "KOI8-U", 6},
      {2088, "csKOI8U", 7},
      {2089, "IBM00858", 8},
      {2089, "CCSID00858", 10},
      {2089, "CP00858", 7},
      {2089, "PC-Multilingual-850+euro", 24},
      {2089, "csIBM00858", 10},
      {2090, "IBM00924", 8},
      {2090, "CCSID00924", 10},
      {2090, "CP00924", 7},
      {2090, "csIBM00924", 10},
      {2090, "ebcdic-Latin9--euro", 19},
      {2091, "IBM01140", 8},
      {2091, "CCSID01140", 10},
      {2091, "CP01140", 7},
      {2091, "csIBM01140", 10},
      {2091, "ebcdic-us-37+euro", 17},
      {2092, "IBM01141", 8},
      {2092, "CCSID01141", 10},
      {2092, "CP01141", 7},
      {2092, "csIBM01141", 10},
      {2092, "ebcdic-de-273+euro", 18},
      {2093, "IBM01142", 8},
      {2093, "CCSID01142", 10},
      {2093, "CP01142", 7},
      {2093, "csIBM01142", 10},
      {2093, "ebcdic-dk-277+euro", 18},
      {2093, "ebcdic-no-277+euro", 18},
      {2094, "IBM01143", 8},
      {2094, "CCSID01143", 10},
      {2094, "CP01143", 7},
      {2094, "csIBM01143", 10},
      {2094, "ebcdic-fi-278+euro", 18},
      {2094, "ebcdic-se-278+euro", 18},
      {2095, "IBM01144", 8},
      {2095, "CCSID01144", 10},
      {2095, "CP01144", 7},
      {2095, "csIBM01144", 10},
      {2095, "ebcdic-it-280+euro", 18},
      {2096, "IBM01145", 8},
      {2096, "CCSID01145", 10},
      {2096, "CP01145", 7},
      {2096, "csIBM01145", 10},
      {2096, "ebcdic-es-284+euro", 18},
      {2097, "IBM01146", 8},
      {2097, "CCSID01146", 10},
      {2097, "CP01146", 7},
      {2097, "csIBM01146", 10},
      {2097, "ebcdic-gb-285+euro", 18},
      {2098, "IBM01147", 8},
      {2098, "CCSID01147", 10},
      {2098, "CP01147", 7},
      {2098, "csIBM01147", 10},
      {2098, "ebcdic-fr-297+euro", 18},
      {2099, "IBM01148", 8},
      {2099, "CCSID01148", 10},
      {2099, "CP01148", 7},
      {2099, "csIBM01148", 10},
      {2099, "ebcdic-international-500+euro", 29},
      {2100, "IBM01149", 8},
      {2100, "CCSID01149", 10},
      {2100, "CP01149", 7},
      {2100, "csIBM01149", 10},
      {2100, "ebcdic-is-871+euro", 18},
      {2101, "Big5-HKSCS", 10},
      {2101, "csBig5HKSCS", 11},
      {2102, "IBM1047", 7},
      {2102, "IBM-1047", 8},
      {2102, "csIBM1047", 9},
      {2103, "PTCP154", 7},
      {2103, "CP154", 5},
      {2103, "Cyrillic-Asian", 14},
      {2103, "PT154", 5},
      {2103, "csPTCP154", 9},
      {2104, "Amiga-1251", 10},
      {2104, "Ami-1251", 8},
      {2104, "Ami1251", 7},
      {2104, "Amiga1251", 9},
      {2104, "csAmiga1251", 11},
      {2105, "KOI7-switched", 13},
      {2105, "csKOI7switched", 14},
      {2106, "BRF", 3},
      {2106, "csBRF", 5},
      {2107, "TSCII", 5},
      {2107, "csTSCII", 7},
      {2108, "CP51932", 7},
      {2108, "csCP51932", 9},
      {2109, "windows-874", 11},
      {2109, "cswindows874", 12},
      {2250, "windows-1250", 12},
      {2250, "cswindows1250", 13},
      {2251, "windows-1251", 12},
      {2251, "cswindows1251", 13},
      {2252, "windows-1252", 12},
      {2252, "cswindows1252", 13},
      {2253, "windows-1253", 12},
      {2253, "cswindows1253", 13},
      {2254, "windows-1254", 12},
      {2254, "cswindows1254", 13},
      {2255, "windows-1255", 12},
      {2255, "cswindows1255", 13},
      {2256, "windows-1256", 12},
      {2256, "cswindows1256", 13},
      {2257, "windows-1257", 12},
      {2257, "cswindows1257", 13},
      {2258, "windows-1258", 12},
      {2258, "cswindows1258", 13},
      {2259, "TIS-620", 7},
      {2259, "ISO-8859-11", 11},
      {2259, "csTIS620", 8},
      {2260, "CP50220", 7},
      {2260, "csCP50220", 9},
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
