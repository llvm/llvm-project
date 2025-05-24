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

#if _LIBCPP_HAS_LOCALIZATION

#include <__algorithm/copy_n.h>
#include <__algorithm/lower_bound.h>
#include <__algorithm/min.h>
#include <__functional/hash.h>
#include <__iterator/iterator_traits.h>
#include <__locale_dir/locale_base_api.h>
#include <__ranges/view_interface.h>
#include <__string/char_traits.h>
#include <__utility/unreachable.h>
#include <cstdint>
#include <string_view>

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 26
_LIBCPP_BEGIN_NAMESPACE_STD

struct _LIBCPP_EXPORTED_FROM_ABI text_encoding {
  static constexpr size_t max_name_length = 63;

private:
  struct __encoding_data {
    using __id_rep _LIBCPP_NODEBUG = int_least32_t;
    __id_rep __mib_rep;
    const char* __name;

    friend constexpr bool operator==(const __encoding_data& __e, const __encoding_data& __other) _NOEXCEPT {
      return __e.__mib_rep == __other.__mib_rep || __comp_name(__e.__name, __other.__name);
    }

    friend constexpr bool operator<(const __encoding_data& __e, const __id_rep __i) _NOEXCEPT {
      return __e.__mib_rep < __i;
    }
  };

public:
  enum class id : __encoding_data::__id_rep {
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
    NATSDANO                = 33,
    NATSDANOADD             = 34,
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
    IBBM904                 = 2060,
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
    CP50220                 = 2260,
    reserved                = 3000
  };

  using enum id;

  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding() = default;
  _LIBCPP_HIDE_FROM_ABI constexpr explicit text_encoding(string_view __enc) _NOEXCEPT
      : __encoding_rep_(__find_encoding_data(__enc)) {
    __enc.copy(__name_, max_name_length, 0);
  }
  _LIBCPP_HIDE_FROM_ABI constexpr text_encoding(id __i) _NOEXCEPT : __encoding_rep_(__find_encoding_data_by_id(__i)) {
    if (__encoding_rep_->__name[0] != '\0')
      std::copy_n(__encoding_rep_->__name, std::char_traits<char>::length(__encoding_rep_->__name), __name_);
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr id mib() const _NOEXCEPT { return id(__encoding_rep_->__mib_rep); }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const char* name() const _NOEXCEPT { return __name_; }

  // [text.encoding.aliases], class text_encoding::aliases_view
  struct aliases_view : ranges::view_interface<aliases_view> {
    constexpr aliases_view() = default;
    constexpr aliases_view(const __encoding_data* __d) : __view_data_(__d) {}
    struct __end_sentinel {};
    struct __iterator {
      using value_type        = const char*;
      using reference         = const char*;
      using difference_type   = ptrdiff_t;

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator() noexcept = default; 
      
      _LIBCPP_HIDE_FROM_ABI constexpr value_type operator*() const {
        if (__can_dereference())
          return __data_->__name;
        std::unreachable();
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

      _LIBCPP_HIDE_FROM_ABI constexpr difference_type operator-(const __iterator& __other) const 
      {
        if(__other.__mib_rep_ == __mib_rep_)
          return __mib_rep_ - __other.__mib_rep_;
        std::unreachable();
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
        if (__data_) [[__likely__]] {
          if (__n > 0) {
            if ((__data_ + __n) < std::end(__text_encoding_data) && __data_[__n - 1].__mib_rep == __mib_rep_)
              __data_ += __n;
            else
              *this = __iterator{};
          } else if (__n < 0) {
            if ((__data_ + __n) > __text_encoding_data && __data_[__n].__mib_rep == __mib_rep_)
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
          : __data_(__enc_d), __mib_rep_(__enc_d ? __enc_d->__mib_rep : 0) {}

      _LIBCPP_HIDE_FROM_ABI bool __can_dereference() const { return __data_ && __data_->__mib_rep == __mib_rep_; }

      // default iterator is a sentinel
      const __encoding_data* __data_       = nullptr;
      __encoding_data::__id_rep __mib_rep_ = 0;
    };

    constexpr __iterator begin() const { return __iterator{__view_data_}; }
    constexpr __end_sentinel end() const { return __end_sentinel{}; }

  private:
    const __encoding_data* __view_data_ = nullptr;
  };

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr aliases_view aliases() const _NOEXCEPT {
    auto __rep = __encoding_rep_ - 1;
    if (__encoding_rep_->__name[0]) {
      while (__rep > std::begin(__text_encoding_data) && (__rep--)->__mib_rep == __encoding_rep_->__mib_rep)
        ;
    } else {
      __rep = nullptr;
    }

    return aliases_view(__rep);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __a, const text_encoding& __b) _NOEXCEPT {
    if (__a.mib() == id::other && __b.mib() == id::other)
      return __comp_name(__a.__name_, __b.__name_);

    return __a.mib() == __b.mib();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const text_encoding& __encoding, id __i) _NOEXCEPT {
    return __encoding.mib() == __i;
  }

#    if __CHAR_BIT__ == 8
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static consteval text_encoding literal() _NOEXCEPT {
#      ifdef __GNUC_EXECUTION_CHARSET_NAME
    return text_encoding(__GNUC_EXECUTION_CHARSET_NAME);
#      elif defined(__clang_literal_encoding__)
    return text_encoding(__clang_literal_encoding__);
#      elif defined(__clang__)
    return text_encoding(id::UTF8);
#      else
    return {};
#      endif
  }

  [[nodiscard]] static text_encoding environment();

  template <id __i>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is() {
    return environment() == __i;
  }

#    else
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static consteval text_encoding literal() = delete;
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static text_encoding environment()       = delete;
  template <id __i>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI static bool environment_is() = delete;
#    endif

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
    auto __data_ptr = __text_encoding_data + 2, __data_last = std::end(__text_encoding_data) - 1;

    for (; __data_ptr != __data_last; __data_ptr++) {
      if (__comp_name(__a, __data_ptr->__name)) {
        const auto __found_id = __data_ptr->__mib_rep;
        while (__data_ptr[-1].__mib_rep == __found_id)
          __data_ptr--;
        return __data_ptr;
      }
    }

    return __text_encoding_data; // other
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr const __encoding_data* __find_encoding_data_by_id(id __i) {
    auto __found = std::lower_bound(
        std::begin(__text_encoding_data), std::end(__text_encoding_data) - 1, __encoding_data::__id_rep(__i));
    return __found != std::end(__text_encoding_data) - 1 ? __found : __text_encoding_data + 1; // unknown
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr __encoding_data __text_encoding_data[] = {
      {1, ""},
      {2, ""},
      {3, "ANSI_X3.4-1968"},
      {3, "ANSI_X3.4-1986"},
      {3, "IBM367"},
      {3, "ISO646-US"},
      {3, "ISO_646.irv:1991"},
      {3, "cp367"},
      {3, "csASCII"},
      {3, "iso-ir-6"},
      {3, "us"},
      {4, "ISO-8859-1"},
      {4, "ISO_8859-1:1987"},
      {4, "CP819"},
      {4, "IBM819"},
      {4, "ISO_8859-1"},
      {4, "csISOLatin1"},
      {4, "iso-ir-100"},
      {4, "l1"},
      {4, "latin1"},
      {5, "ISO-8859-2"},
      {5, "ISO_8859-2:1987"},
      {5, "ISO_8859-2"},
      {5, "csISOLatin2"},
      {5, "iso-ir-101"},
      {5, "l2"},
      {5, "latin2"},
      {6, "ISO-8859-3"},
      {6, "ISO_8859-3:1988"},
      {6, "ISO_8859-3"},
      {6, "csISOLatin3"},
      {6, "iso-ir-109"},
      {6, "l3"},
      {6, "latin3"},
      {7, "ISO-8859-4"},
      {7, "ISO_8859-4:1988"},
      {7, "ISO_8859-4"},
      {7, "csISOLatin4"},
      {7, "iso-ir-110"},
      {7, "l4"},
      {7, "latin4"},
      {8, "ISO-8859-5"},
      {8, "ISO_8859-5:1988"},
      {8, "ISO_8859-5"},
      {8, "csISOLatinCyrillic"},
      {8, "cyrillic"},
      {8, "iso-ir-144"},
      {9, "ISO-8859-6"},
      {9, "ISO_8859-6:1987"},
      {9, "ASMO-708"},
      {9, "ECMA-114"},
      {9, "ISO_8859-6"},
      {9, "arabic"},
      {9, "csISOLatinArabic"},
      {9, "iso-ir-127"},
      {10, "ISO-8859-7"},
      {10, "ISO_8859-7:1987"},
      {10, "ECMA-118"},
      {10, "ELOT_928"},
      {10, "ISO_8859-7"},
      {10, "csISOLatinGreek"},
      {10, "greek"},
      {10, "greek8"},
      {10, "iso-ir-126"},
      {11, "ISO-8859-8"},
      {11, "ISO_8859-8:1988"},
      {11, "ISO_8859-8"},
      {11, "csISOLatinHebrew"},
      {11, "hebrew"},
      {11, "iso-ir-138"},
      {12, "ISO-8859-9"},
      {12, "ISO_8859-9:1989"},
      {12, "ISO_8859-9"},
      {12, "csISOLatin5"},
      {12, "iso-ir-148"},
      {12, "l5"},
      {12, "latin5"},
      {13, "ISO-8859-10"},
      {13, "ISO_8859-10:1992"},
      {13, "csISOLatin6"},
      {13, "iso-ir-157"},
      {13, "l6"},
      {13, "latin6"},
      {14, "ISO_6937-2-add"},
      {14, "csISOTextComm"},
      {14, "iso-ir-142"},
      {15, "JIS_X0201"},
      {15, "X0201"},
      {15, "csHalfWidthKatakana"},
      {16, "JIS_Encoding"},
      {16, "csJISEncoding"},
      {17, "Shift_JIS"},
      {17, "MS_Kanji"},
      {17, "csShiftJIS"},
      {18, "EUC-JP"},
      {18, "Extended_UNIX_Code_Packed_Format_for_Japanese"},
      {18, "csEUCPkdFmtJapanese"},
      {19, "Extended_UNIX_Code_Fixed_Width_for_Japanese"},
      {19, "csEUCFixWidJapanese"},
      {20, "BS_4730"},
      {20, "ISO646-GB"},
      {20, "csISO4UnitedKingdom"},
      {20, "gb"},
      {20, "iso-ir-4"},
      {20, "uk"},
      {21, "SEN_850200_C"},
      {21, "ISO646-SE2"},
      {21, "csISO11SwedishForNames"},
      {21, "iso-ir-11"},
      {21, "se2"},
      {22, "IT"},
      {22, "ISO646-IT"},
      {22, "csISO15Italian"},
      {22, "iso-ir-15"},
      {23, "ES"},
      {23, "ISO646-ES"},
      {23, "csISO17Spanish"},
      {23, "iso-ir-17"},
      {24, "DIN_66003"},
      {24, "ISO646-DE"},
      {24, "csISO21German"},
      {24, "de"},
      {24, "iso-ir-21"},
      {25, "NS_4551-1"},
      {25, "ISO646-NO"},
      {25, "csISO60DanishNorwegian"},
      {25, "csISO60Norwegian1"},
      {25, "iso-ir-60"},
      {25, "no"},
      {26, "NF_Z_62-010"},
      {26, "ISO646-FR"},
      {26, "csISO69French"},
      {26, "fr"},
      {26, "iso-ir-69"},
      {27, "ISO-10646-UTF-1"},
      {27, "csISO10646UTF1"},
      {28, "ISO_646.basic:1983"},
      {28, "csISO646basic1983"},
      {28, "ref"},
      {29, "INVARIANT"},
      {29, "csINVARIANT"},
      {30, "ISO_646.irv:1983"},
      {30, "csISO2IntlRefVersion"},
      {30, "irv"},
      {30, "iso-ir-2"},
      {31, "NATS-SEFI"},
      {31, "csNATSSEFI"},
      {31, "iso-ir-8-1"},
      {32, "NATS-SEFI-ADD"},
      {32, "csNATSSEFIADD"},
      {32, "iso-ir-8-2"},
      {33, "NATS-DANO"},
      {33, "csNATSDANO"},
      {33, "iso-ir-9-1"},
      {34, "NATS-DANO-ADD"},
      {34, "csNATSDANOADD"},
      {34, "iso-ir-9-2"},
      {35, "SEN_850200_B"},
      {35, "FI"},
      {35, "ISO646-FI"},
      {35, "ISO646-SE"},
      {35, "csISO10Swedish"},
      {35, "iso-ir-10"},
      {35, "se"},
      {36, "KS_C_5601-1987"},
      {36, "KSC_5601"},
      {36, "KS_C_5601-1989"},
      {36, "csKSC56011987"},
      {36, "iso-ir-149"},
      {36, "korean"},
      {37, "ISO-2022-KR"},
      {37, "csISO2022KR"},
      {38, "EUC-KR"},
      {38, "csEUCKR"},
      {39, "ISO-2022-JP"},
      {39, "csISO2022JP"},
      {40, "ISO-2022-JP-2"},
      {40, "csISO2022JP2"},
      {41, "JIS_C6220-1969-jp"},
      {41, "JIS_C6220-1969"},
      {41, "csISO13JISC6220jp"},
      {41, "iso-ir-13"},
      {41, "katakana"},
      {41, "x0201-7"},
      {42, "JIS_C6220-1969-ro"},
      {42, "ISO646-JP"},
      {42, "csISO14JISC6220ro"},
      {42, "iso-ir-14"},
      {42, "jp"},
      {43, "PT"},
      {43, "ISO646-PT"},
      {43, "csISO16Portuguese"},
      {43, "iso-ir-16"},
      {44, "greek7-old"},
      {44, "csISO18Greek7Old"},
      {44, "iso-ir-18"},
      {45, "latin-greek"},
      {45, "csISO19LatinGreek"},
      {45, "iso-ir-19"},
      {46, "NF_Z_62-010_(1973)"},
      {46, "ISO646-FR1"},
      {46, "csISO25French"},
      {46, "iso-ir-25"},
      {47, "Latin-greek-1"},
      {47, "csISO27LatinGreek1"},
      {47, "iso-ir-27"},
      {48, "ISO_5427"},
      {48, "csISO5427Cyrillic"},
      {48, "iso-ir-37"},
      {49, "JIS_C6226-1978"},
      {49, "csISO42JISC62261978"},
      {49, "iso-ir-42"},
      {50, "BS_viewdata"},
      {50, "csISO47BSViewdata"},
      {50, "iso-ir-47"},
      {51, "INIS"},
      {51, "csISO49INIS"},
      {51, "iso-ir-49"},
      {52, "INIS-8"},
      {52, "csISO50INIS8"},
      {52, "iso-ir-50"},
      {53, "INIS-cyrillic"},
      {53, "csISO51INISCyrillic"},
      {53, "iso-ir-51"},
      {54, "ISO_5427:1981"},
      {54, "ISO5427Cyrillic1981"},
      {54, "csISO54271981"},
      {54, "iso-ir-54"},
      {55, "ISO_5428:1980"},
      {55, "csISO5428Greek"},
      {55, "iso-ir-55"},
      {56, "GB_1988-80"},
      {56, "ISO646-CN"},
      {56, "cn"},
      {56, "csISO57GB1988"},
      {56, "iso-ir-57"},
      {57, "GB_2312-80"},
      {57, "chinese"},
      {57, "csISO58GB231280"},
      {57, "iso-ir-58"},
      {58, "NS_4551-2"},
      {58, "ISO646-NO2"},
      {58, "csISO61Norwegian2"},
      {58, "iso-ir-61"},
      {58, "no2"},
      {59, "videotex-suppl"},
      {59, "csISO70VideotexSupp1"},
      {59, "iso-ir-70"},
      {60, "PT2"},
      {60, "ISO646-PT2"},
      {60, "csISO84Portuguese2"},
      {60, "iso-ir-84"},
      {61, "ES2"},
      {61, "ISO646-ES2"},
      {61, "csISO85Spanish2"},
      {61, "iso-ir-85"},
      {62, "MSZ_7795.3"},
      {62, "ISO646-HU"},
      {62, "csISO86Hungarian"},
      {62, "hu"},
      {62, "iso-ir-86"},
      {63, "JIS_C6226-1983"},
      {63, "JIS_X0208-1983"},
      {63, "csISO87JISX0208"},
      {63, "iso-ir-87"},
      {63, "x0208"},
      {64, "greek7"},
      {64, "csISO88Greek7"},
      {64, "iso-ir-88"},
      {65, "ASMO_449"},
      {65, "ISO_9036"},
      {65, "arabic7"},
      {65, "csISO89ASMO449"},
      {65, "iso-ir-89"},
      {66, "iso-ir-90"},
      {66, "csISO90"},
      {67, "JIS_C6229-1984-a"},
      {67, "csISO91JISC62291984a"},
      {67, "iso-ir-91"},
      {67, "jp-ocr-a"},
      {68, "JIS_C6229-1984-b"},
      {68, "ISO646-JP-OCR-B"},
      {68, "csISO92JISC62991984b"},
      {68, "iso-ir-92"},
      {68, "jp-ocr-b"},
      {69, "JIS_C6229-1984-b-add"},
      {69, "csISO93JIS62291984badd"},
      {69, "iso-ir-93"},
      {69, "jp-ocr-b-add"},
      {70, "JIS_C6229-1984-hand"},
      {70, "csISO94JIS62291984hand"},
      {70, "iso-ir-94"},
      {70, "jp-ocr-hand"},
      {71, "JIS_C6229-1984-hand-add"},
      {71, "csISO95JIS62291984handadd"},
      {71, "iso-ir-95"},
      {71, "jp-ocr-hand-add"},
      {72, "JIS_C6229-1984-kana"},
      {72, "csISO96JISC62291984kana"},
      {72, "iso-ir-96"},
      {73, "ISO_2033-1983"},
      {73, "csISO2033"},
      {73, "e13b"},
      {73, "iso-ir-98"},
      {74, "ANSI_X3.110-1983"},
      {74, "CSA_T500-1983"},
      {74, "NAPLPS"},
      {74, "csISO99NAPLPS"},
      {74, "iso-ir-99"},
      {75, "T.61-7bit"},
      {75, "csISO102T617bit"},
      {75, "iso-ir-102"},
      {76, "T.61-8bit"},
      {76, "T.61"},
      {76, "csISO103T618bit"},
      {76, "iso-ir-103"},
      {77, "ECMA-cyrillic"},
      {77, "KOI8-E"},
      {77, "csISO111ECMACyrillic"},
      {77, "iso-ir-111"},
      {78, "CSA_Z243.4-1985-1"},
      {78, "ISO646-CA"},
      {78, "ca"},
      {78, "csISO121Canadian1"},
      {78, "csa7-1"},
      {78, "csa71"},
      {78, "iso-ir-121"},
      {79, "CSA_Z243.4-1985-2"},
      {79, "ISO646-CA2"},
      {79, "csISO122Canadian2"},
      {79, "csa7-2"},
      {79, "csa72"},
      {79, "iso-ir-122"},
      {80, "CSA_Z243.4-1985-gr"},
      {80, "csISO123CSAZ24341985gr"},
      {80, "iso-ir-123"},
      {81, "ISO-8859-6-E"},
      {81, "ISO_8859-6-E"},
      {81, "csISO88596E"},
      {82, "ISO-8859-6-I"},
      {82, "ISO_8859-6-I"},
      {82, "csISO88596I"},
      {83, "T.101-G2"},
      {83, "csISO128T101G2"},
      {83, "iso-ir-128"},
      {84, "ISO-8859-8-E"},
      {84, "ISO_8859-8-E"},
      {84, "csISO88598E"},
      {85, "ISO-8859-8-I"},
      {85, "ISO_8859-8-I"},
      {85, "csISO88598I"},
      {86, "CSN_369103"},
      {86, "csISO139CSN369103"},
      {86, "iso-ir-139"},
      {87, "JUS_I.B1.002"},
      {87, "ISO646-YU"},
      {87, "csISO141JUSIB1002"},
      {87, "iso-ir-141"},
      {87, "js"},
      {87, "yu"},
      {88, "IEC_P27-1"},
      {88, "csISO143IECP271"},
      {88, "iso-ir-143"},
      {89, "JUS_I.B1.003-serb"},
      {89, "csISO146Serbian"},
      {89, "iso-ir-146"},
      {89, "serbian"},
      {90, "JUS_I.B1.003-mac"},
      {90, "csISO147Macedonian"},
      {90, "iso-ir-147"},
      {90, "macedonian"},
      {91, "greek-ccitt"},
      {91, "csISO150"},
      {91, "csISO150GreekCCITT"},
      {91, "iso-ir-150"},
      {92, "NC_NC00-10:81"},
      {92, "ISO646-CU"},
      {92, "csISO151Cuba"},
      {92, "cuba"},
      {92, "iso-ir-151"},
      {93, "ISO_6937-2-25"},
      {93, "csISO6937Add"},
      {93, "iso-ir-152"},
      {94, "GOST_19768-74"},
      {94, "ST_SEV_358-88"},
      {94, "csISO153GOST1976874"},
      {94, "iso-ir-153"},
      {95, "ISO_8859-supp"},
      {95, "csISO8859Supp"},
      {95, "iso-ir-154"},
      {95, "latin1-2-5"},
      {96, "ISO_10367-box"},
      {96, "csISO10367Box"},
      {96, "iso-ir-155"},
      {97, "latin-lap"},
      {97, "csISO158Lap"},
      {97, "iso-ir-158"},
      {97, "lap"},
      {98, "JIS_X0212-1990"},
      {98, "csISO159JISX02121990"},
      {98, "iso-ir-159"},
      {98, "x0212"},
      {99, "DS_2089"},
      {99, "DS2089"},
      {99, "ISO646-DK"},
      {99, "csISO646Danish"},
      {99, "dk"},
      {100, "us-dk"},
      {100, "csUSDK"},
      {101, "dk-us"},
      {101, "csDKUS"},
      {102, "KSC5636"},
      {102, "ISO646-KR"},
      {102, "csKSC5636"},
      {103, "UNICODE-1-1-UTF-7"},
      {103, "csUnicode11UTF7"},
      {104, "ISO-2022-CN"},
      {104, "csISO2022CN"},
      {105, "ISO-2022-CN-EXT"},
      {105, "csISO2022CNEXT"},
      {106, "UTF-8"},
      {106, "csUTF8"},
      {109, "ISO-8859-13"},
      {109, "csISO885913"},
      {110, "ISO-8859-14"},
      {110, "ISO_8859-14"},
      {110, "ISO_8859-14:1998"},
      {110, "csISO885914"},
      {110, "iso-celtic"},
      {110, "iso-ir-199"},
      {110, "l8"},
      {110, "latin8"},
      {111, "ISO-8859-15"},
      {111, "ISO_8859-15"},
      {111, "Latin-9"},
      {111, "csISO885915"},
      {112, "ISO-8859-16"},
      {112, "ISO_8859-16"},
      {112, "ISO_8859-16:2001"},
      {112, "csISO885916"},
      {112, "iso-ir-226"},
      {112, "l10"},
      {112, "latin10"},
      {113, "GBK"},
      {113, "CP936"},
      {113, "MS936"},
      {113, "csGBK"},
      {113, "windows-936"},
      {114, "GB18030"},
      {114, "csGB18030"},
      {115, "OSD_EBCDIC_DF04_15"},
      {115, "csOSDEBCDICDF0415"},
      {116, "OSD_EBCDIC_DF03_IRV"},
      {116, "csOSDEBCDICDF03IRV"},
      {117, "OSD_EBCDIC_DF04_1"},
      {117, "csOSDEBCDICDF041"},
      {118, "ISO-11548-1"},
      {118, "ISO_11548-1"},
      {118, "ISO_TR_11548-1"},
      {118, "csISO115481"},
      {119, "KZ-1048"},
      {119, "RK1048"},
      {119, "STRK1048-2002"},
      {119, "csKZ1048"},
      {1000, "ISO-10646-UCS-2"},
      {1000, "csUnicode"},
      {1001, "ISO-10646-UCS-4"},
      {1001, "csUCS4"},
      {1002, "ISO-10646-UCS-Basic"},
      {1002, "csUnicodeASCII"},
      {1003, "ISO-10646-Unicode-Latin1"},
      {1003, "ISO-10646"},
      {1003, "csUnicodeLatin1"},
      {1004, "ISO-10646-J-1"},
      {1004, "csUnicodeJapanese"},
      {1005, "ISO-Unicode-IBM-1261"},
      {1005, "csUnicodeIBM1261"},
      {1006, "ISO-Unicode-IBM-1268"},
      {1006, "csUnicodeIBM1268"},
      {1007, "ISO-Unicode-IBM-1276"},
      {1007, "csUnicodeIBM1276"},
      {1008, "ISO-Unicode-IBM-1264"},
      {1008, "csUnicodeIBM1264"},
      {1009, "ISO-Unicode-IBM-1265"},
      {1009, "csUnicodeIBM1265"},
      {1010, "UNICODE-1-1"},
      {1010, "csUnicode11"},
      {1011, "SCSU"},
      {1011, "csSCSU"},
      {1012, "UTF-7"},
      {1012, "csUTF7"},
      {1013, "UTF-16BE"},
      {1013, "csUTF16BE"},
      {1014, "UTF-16LE"},
      {1014, "csUTF16LE"},
      {1015, "UTF-16"},
      {1015, "csUTF16"},
      {1016, "CESU-8"},
      {1016, "csCESU-8"},
      {1016, "csCESU8"},
      {1017, "UTF-32"},
      {1017, "csUTF32"},
      {1018, "UTF-32BE"},
      {1018, "csUTF32BE"},
      {1019, "UTF-32LE"},
      {1019, "csUTF32LE"},
      {1020, "BOCU-1"},
      {1020, "csBOCU-1"},
      {1020, "csBOCU1"},
      {1021, "UTF-7-IMAP"},
      {1021, "csUTF7IMAP"},
      {2000, "ISO-8859-1-Windows-3.0-Latin-1"},
      {2000, "csWindows30Latin1"},
      {2001, "ISO-8859-1-Windows-3.1-Latin-1"},
      {2001, "csWindows31Latin1"},
      {2002, "ISO-8859-2-Windows-Latin-2"},
      {2002, "csWindows31Latin2"},
      {2003, "ISO-8859-9-Windows-Latin-5"},
      {2003, "csWindows31Latin5"},
      {2004, "hp-roman8"},
      {2004, "csHPRoman8"},
      {2004, "r8"},
      {2004, "roman8"},
      {2005, "Adobe-Standard-Encoding"},
      {2005, "csAdobeStandardEncoding"},
      {2006, "Ventura-US"},
      {2006, "csVenturaUS"},
      {2007, "Ventura-International"},
      {2007, "csVenturaInternational"},
      {2008, "DEC-MCS"},
      {2008, "csDECMCS"},
      {2008, "dec"},
      {2009, "IBM850"},
      {2009, "850"},
      {2009, "cp850"},
      {2009, "csPC850Multilingual"},
      {2010, "IBM852"},
      {2010, "852"},
      {2010, "cp852"},
      {2010, "csPCp852"},
      {2011, "IBM437"},
      {2011, "437"},
      {2011, "cp437"},
      {2011, "csPC8CodePage437"},
      {2012, "PC8-Danish-Norwegian"},
      {2012, "csPC8DanishNorwegian"},
      {2013, "IBM862"},
      {2013, "862"},
      {2013, "cp862"},
      {2013, "csPC862LatinHebrew"},
      {2014, "PC8-Turkish"},
      {2014, "csPC8Turkish"},
      {2015, "IBM-Symbols"},
      {2015, "csIBMSymbols"},
      {2016, "IBM-Thai"},
      {2016, "csIBMThai"},
      {2017, "HP-Legal"},
      {2017, "csHPLegal"},
      {2018, "HP-Pi-font"},
      {2018, "csHPPiFont"},
      {2019, "HP-Math8"},
      {2019, "csHPMath8"},
      {2020, "Adobe-Symbol-Encoding"},
      {2020, "csHPPSMath"},
      {2021, "HP-DeskTop"},
      {2021, "csHPDesktop"},
      {2022, "Ventura-Math"},
      {2022, "csVenturaMath"},
      {2023, "Microsoft-Publishing"},
      {2023, "csMicrosoftPublishing"},
      {2024, "Windows-31J"},
      {2024, "csWindows31J"},
      {2025, "GB2312"},
      {2025, "csGB2312"},
      {2026, "Big5"},
      {2026, "csBig5"},
      {2027, "macintosh"},
      {2027, "csMacintosh"},
      {2027, "mac"},
      {2028, "IBM037"},
      {2028, "cp037"},
      {2028, "csIBM037"},
      {2028, "ebcdic-cp-ca"},
      {2028, "ebcdic-cp-nl"},
      {2028, "ebcdic-cp-us"},
      {2028, "ebcdic-cp-wt"},
      {2029, "IBM038"},
      {2029, "EBCDIC-INT"},
      {2029, "cp038"},
      {2029, "csIBM038"},
      {2030, "IBM273"},
      {2030, "CP273"},
      {2030, "csIBM273"},
      {2031, "IBM274"},
      {2031, "CP274"},
      {2031, "EBCDIC-BE"},
      {2031, "csIBM274"},
      {2032, "IBM275"},
      {2032, "EBCDIC-BR"},
      {2032, "cp275"},
      {2032, "csIBM275"},
      {2033, "IBM277"},
      {2033, "EBCDIC-CP-DK"},
      {2033, "EBCDIC-CP-NO"},
      {2033, "csIBM277"},
      {2034, "IBM278"},
      {2034, "CP278"},
      {2034, "csIBM278"},
      {2034, "ebcdic-cp-fi"},
      {2034, "ebcdic-cp-se"},
      {2035, "IBM280"},
      {2035, "CP280"},
      {2035, "csIBM280"},
      {2035, "ebcdic-cp-it"},
      {2036, "IBM281"},
      {2036, "EBCDIC-JP-E"},
      {2036, "cp281"},
      {2036, "csIBM281"},
      {2037, "IBM284"},
      {2037, "CP284"},
      {2037, "csIBM284"},
      {2037, "ebcdic-cp-es"},
      {2038, "IBM285"},
      {2038, "CP285"},
      {2038, "csIBM285"},
      {2038, "ebcdic-cp-gb"},
      {2039, "IBM290"},
      {2039, "EBCDIC-JP-kana"},
      {2039, "cp290"},
      {2039, "csIBM290"},
      {2040, "IBM297"},
      {2040, "cp297"},
      {2040, "csIBM297"},
      {2040, "ebcdic-cp-fr"},
      {2041, "IBM420"},
      {2041, "cp420"},
      {2041, "csIBM420"},
      {2041, "ebcdic-cp-ar1"},
      {2042, "IBM423"},
      {2042, "cp423"},
      {2042, "csIBM423"},
      {2042, "ebcdic-cp-gr"},
      {2043, "IBM424"},
      {2043, "cp424"},
      {2043, "csIBM424"},
      {2043, "ebcdic-cp-he"},
      {2044, "IBM500"},
      {2044, "CP500"},
      {2044, "csIBM500"},
      {2044, "ebcdic-cp-be"},
      {2044, "ebcdic-cp-ch"},
      {2045, "IBM851"},
      {2045, "851"},
      {2045, "cp851"},
      {2045, "csIBM851"},
      {2046, "IBM855"},
      {2046, "855"},
      {2046, "cp855"},
      {2046, "csIBM855"},
      {2047, "IBM857"},
      {2047, "857"},
      {2047, "cp857"},
      {2047, "csIBM857"},
      {2048, "IBM860"},
      {2048, "860"},
      {2048, "cp860"},
      {2048, "csIBM860"},
      {2049, "IBM861"},
      {2049, "861"},
      {2049, "cp-is"},
      {2049, "cp861"},
      {2049, "csIBM861"},
      {2050, "IBM863"},
      {2050, "863"},
      {2050, "cp863"},
      {2050, "csIBM863"},
      {2051, "IBM864"},
      {2051, "cp864"},
      {2051, "csIBM864"},
      {2052, "IBM865"},
      {2052, "865"},
      {2052, "cp865"},
      {2052, "csIBM865"},
      {2053, "IBM868"},
      {2053, "CP868"},
      {2053, "cp-ar"},
      {2053, "csIBM868"},
      {2054, "IBM869"},
      {2054, "869"},
      {2054, "cp-gr"},
      {2054, "cp869"},
      {2054, "csIBM869"},
      {2055, "IBM870"},
      {2055, "CP870"},
      {2055, "csIBM870"},
      {2055, "ebcdic-cp-roece"},
      {2055, "ebcdic-cp-yu"},
      {2056, "IBM871"},
      {2056, "CP871"},
      {2056, "csIBM871"},
      {2056, "ebcdic-cp-is"},
      {2057, "IBM880"},
      {2057, "EBCDIC-Cyrillic"},
      {2057, "cp880"},
      {2057, "csIBM880"},
      {2058, "IBM891"},
      {2058, "cp891"},
      {2058, "csIBM891"},
      {2059, "IBM903"},
      {2059, "cp903"},
      {2059, "csIBM903"},
      {2060, "IBM904"},
      {2060, "904"},
      {2060, "cp904"},
      {2060, "csIBBM904"},
      {2061, "IBM905"},
      {2061, "CP905"},
      {2061, "csIBM905"},
      {2061, "ebcdic-cp-tr"},
      {2062, "IBM918"},
      {2062, "CP918"},
      {2062, "csIBM918"},
      {2062, "ebcdic-cp-ar2"},
      {2063, "IBM1026"},
      {2063, "CP1026"},
      {2063, "csIBM1026"},
      {2064, "EBCDIC-AT-DE"},
      {2064, "csIBMEBCDICATDE"},
      {2065, "EBCDIC-AT-DE-A"},
      {2065, "csEBCDICATDEA"},
      {2066, "EBCDIC-CA-FR"},
      {2066, "csEBCDICCAFR"},
      {2067, "EBCDIC-DK-NO"},
      {2067, "csEBCDICDKNO"},
      {2068, "EBCDIC-DK-NO-A"},
      {2068, "csEBCDICDKNOA"},
      {2069, "EBCDIC-FI-SE"},
      {2069, "csEBCDICFISE"},
      {2070, "EBCDIC-FI-SE-A"},
      {2070, "csEBCDICFISEA"},
      {2071, "EBCDIC-FR"},
      {2071, "csEBCDICFR"},
      {2072, "EBCDIC-IT"},
      {2072, "csEBCDICIT"},
      {2073, "EBCDIC-PT"},
      {2073, "csEBCDICPT"},
      {2074, "EBCDIC-ES"},
      {2074, "csEBCDICES"},
      {2075, "EBCDIC-ES-A"},
      {2075, "csEBCDICESA"},
      {2076, "EBCDIC-ES-S"},
      {2076, "csEBCDICESS"},
      {2077, "EBCDIC-UK"},
      {2077, "csEBCDICUK"},
      {2078, "EBCDIC-US"},
      {2078, "csEBCDICUS"},
      {2079, "UNKNOWN-8BIT"},
      {2079, "csUnknown8BiT"},
      {2080, "MNEMONIC"},
      {2080, "csMnemonic"},
      {2081, "MNEM"},
      {2081, "csMnem"},
      {2082, "VISCII"},
      {2082, "csVISCII"},
      {2083, "VIQR"},
      {2083, "csVIQR"},
      {2084, "KOI8-R"},
      {2084, "csKOI8R"},
      {2085, "HZ-GB-2312"},
      {2086, "IBM866"},
      {2086, "866"},
      {2086, "cp866"},
      {2086, "csIBM866"},
      {2087, "IBM775"},
      {2087, "cp775"},
      {2087, "csPC775Baltic"},
      {2088, "KOI8-U"},
      {2088, "csKOI8U"},
      {2089, "IBM00858"},
      {2089, "CCSID00858"},
      {2089, "CP00858"},
      {2089, "PC-Multilingual-850+euro"},
      {2089, "csIBM00858"},
      {2090, "IBM00924"},
      {2090, "CCSID00924"},
      {2090, "CP00924"},
      {2090, "csIBM00924"},
      {2090, "ebcdic-Latin9--euro"},
      {2091, "IBM01140"},
      {2091, "CCSID01140"},
      {2091, "CP01140"},
      {2091, "csIBM01140"},
      {2091, "ebcdic-us-37+euro"},
      {2092, "IBM01141"},
      {2092, "CCSID01141"},
      {2092, "CP01141"},
      {2092, "csIBM01141"},
      {2092, "ebcdic-de-273+euro"},
      {2093, "IBM01142"},
      {2093, "CCSID01142"},
      {2093, "CP01142"},
      {2093, "csIBM01142"},
      {2093, "ebcdic-dk-277+euro"},
      {2093, "ebcdic-no-277+euro"},
      {2094, "IBM01143"},
      {2094, "CCSID01143"},
      {2094, "CP01143"},
      {2094, "csIBM01143"},
      {2094, "ebcdic-fi-278+euro"},
      {2094, "ebcdic-se-278+euro"},
      {2095, "IBM01144"},
      {2095, "CCSID01144"},
      {2095, "CP01144"},
      {2095, "csIBM01144"},
      {2095, "ebcdic-it-280+euro"},
      {2096, "IBM01145"},
      {2096, "CCSID01145"},
      {2096, "CP01145"},
      {2096, "csIBM01145"},
      {2096, "ebcdic-es-284+euro"},
      {2097, "IBM01146"},
      {2097, "CCSID01146"},
      {2097, "CP01146"},
      {2097, "csIBM01146"},
      {2097, "ebcdic-gb-285+euro"},
      {2098, "IBM01147"},
      {2098, "CCSID01147"},
      {2098, "CP01147"},
      {2098, "csIBM01147"},
      {2098, "ebcdic-fr-297+euro"},
      {2099, "IBM01148"},
      {2099, "CCSID01148"},
      {2099, "CP01148"},
      {2099, "csIBM01148"},
      {2099, "ebcdic-international-500+euro"},
      {2100, "IBM01149"},
      {2100, "CCSID01149"},
      {2100, "CP01149"},
      {2100, "csIBM01149"},
      {2100, "ebcdic-is-871+euro"},
      {2101, "Big5-HKSCS"},
      {2101, "csBig5HKSCS"},
      {2102, "IBM1047"},
      {2102, "IBM-1047"},
      {2102, "csIBM1047"},
      {2103, "PTCP154"},
      {2103, "CP154"},
      {2103, "Cyrillic-Asian"},
      {2103, "PT154"},
      {2103, "csPTCP154"},
      {2104, "Amiga-1251"},
      {2104, "Ami-1251"},
      {2104, "Ami1251"},
      {2104, "Amiga1251"},
      {2104, "csAmiga1251"},
      {2105, "KOI7-switched"},
      {2105, "csKOI7switched"},
      {2106, "BRF"},
      {2106, "csBRF"},
      {2107, "TSCII"},
      {2107, "csTSCII"},
      {2108, "CP51932"},
      {2108, "csCP51932"},
      {2109, "windows-874"},
      {2109, "cswindows874"},
      {2250, "windows-1250"},
      {2250, "cswindows1250"},
      {2251, "windows-1251"},
      {2251, "cswindows1251"},
      {2252, "windows-1252"},
      {2252, "cswindows1252"},
      {2253, "windows-1253"},
      {2253, "cswindows1253"},
      {2254, "windows-1254"},
      {2254, "cswindows1254"},
      {2255, "windows-1255"},
      {2255, "cswindows1255"},
      {2256, "windows-1256"},
      {2256, "cswindows1256"},
      {2257, "windows-1257"},
      {2257, "cswindows1257"},
      {2258, "windows-1258"},
      {2258, "cswindows1258"},
      {2259, "TIS-620"},
      {2259, "ISO-8859-11"},
      {2259, "csTIS620"},
      {2260, "CP50220"},
      {2260, "csCP50220"},
      {0, nullptr} // sentinel
  };

  const __encoding_data* __encoding_rep_ = __text_encoding_data + 1;
  char __name_[max_name_length + 1]      = {0};
};

template <>
struct hash<text_encoding> {
  size_t operator()(const text_encoding& __enc) const noexcept { return std::hash<text_encoding::id>()(__enc.mib()); }
};

namespace ranges {

template <>
inline constexpr bool enable_borrowed_range<text_encoding::aliases_view> = true;

} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26

_LIBCPP_POP_MACROS

#endif // _LIBCPP_HAS_LOCALIZATION

#endif // _LIBCPP___TEXT_ENCODING_TEXT_ENCODING_H
