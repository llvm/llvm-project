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

#include <__algorithm/find_if.h>
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

  static constexpr int __NATS_DANO_     = 33;
  static constexpr int __NATS_DANO_ADD_ = 34;

  using enum __id;
  static constexpr size_t __max_name_length_ = 63;
  struct __te_data {
    unsigned short __first_alias_index_;
    unsigned short __mib_rep_;
    unsigned char __num_aliases_;

    _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __te_data& __enc, __id __i) noexcept {
      return __enc.__mib_rep_ < static_cast<unsigned short>(__i);
    }

    _LIBCPP_HIDE_FROM_ABI constexpr const char* const* __aliases_begin() const {
      return &__aliases_table[__first_alias_index_];
    }

    _LIBCPP_HIDE_FROM_ABI constexpr const char* const* __aliases_end() const {
      return &__aliases_table[__first_alias_index_ + __num_aliases_];
    }
  };

  _LIBCPP_HIDE_FROM_ABI static constexpr bool __comp_name(string_view __a, string_view __b) {
    if (__a.empty() || __b.empty()) {
      return false;
    }

    // Map any non-alphanumeric character to 255, skip prefix 0s, else get tolower(__n).
    auto __map_char = [](char __n, bool& __in_number) -> unsigned char {
      if (__n == '0') {
        return __in_number ? '0' : 255;
      }
      __in_number = __n >= '1' && __n <= '9';

      if ((__n >= '1' && __n <= '9') || (__n >= 'a' && __n <= 'z')) {
        return __n;
      }
      if (__n >= 'A' && __n <= 'Z') {
        return __n + ('a' - 'A'); // tolower
      }

      return 255;
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

  _LIBCPP_HIDE_FROM_ABI static constexpr unsigned short __find_data_idx(string_view __a) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(__a.size() <= __max_name_length_, "input string_view must have size <= 63");
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(!__a.contains('\0'), "input string_view must not contain '\\0'");

    auto __pred = [&__a](const __te_data& __entry) -> bool {
      // Search slice of alias table that corresponds to the current MIB
      return std::find_if(__entry.__aliases_begin(), __entry.__aliases_end(), [&__a](const char* __alias) {
               return __comp_name(__a, __alias);
             }) != __entry.__aliases_end();
    };

    const __te_data* __found = std::find_if(__entries + 2, std::end(__entries), __pred);
    if (__found == std::end(__entries)) {
      return 0u; // other
    }

    return __found - __entries;
  }

  _LIBCPP_HIDE_FROM_ABI static constexpr unsigned short __find_data_idx_by_id(__id __i) {
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(__i >= __id::other, "invalid text_encoding::id passed");
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(__i <= __id::CP50220, "invalid text_encoding::id passed");
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(int_least32_t(__i) != __NATS_DANO_, "Mib for NATS-DANO used");
    _LIBCPP_ASSERT_ARGUMENT_WITHIN_DOMAIN(int_least32_t(__i) != __NATS_DANO_ADD_, "Mib for NATS-DANO-ADD used");
    auto __found = std::lower_bound(std::begin(__entries), std::end(__entries), __i);

    if (__found == std::end(__entries)) {
      return 1u; // unknown
    }

    return __found - __entries;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __te_impl() = default;

  _LIBCPP_HIDE_FROM_ABI constexpr explicit __te_impl(string_view __enc) noexcept
      : __encoding_idx_(__find_data_idx(__enc)) {
    __enc.copy(__name_, __max_name_length_, 0);
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __te_impl(__id __i) noexcept : __encoding_idx_(__find_data_idx_by_id(__i)) {
    const char* __alias = __aliases_table[__entries[__encoding_idx_].__first_alias_index_];
    if (__alias[0] != '\0') {
      string_view(__alias).copy(__name_, __max_name_length_);
    }
  }

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr __id __mib() const noexcept {
    return __id(__entries[__encoding_idx_].__mib_rep_);
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

      _LIBCPP_HIDE_FROM_ABI constexpr reference operator*() const { return *__data_; }

      _LIBCPP_HIDE_FROM_ABI constexpr reference operator[](difference_type __n) const {
        auto __it = *this;
        return *(__it + __n);
      }

      _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator->() const { return __data_; }

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

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n) { return (*this += -__n); }

      _LIBCPP_HIDE_FROM_ABI constexpr bool operator==(const __iterator& __it) const { return __data_ == __it.__data_; }

      _LIBCPP_HIDE_FROM_ABI constexpr auto operator<=>(__iterator __it) const { return __data_ <=> __it.__data_; }

    private:
      friend struct __aliases_view;

      _LIBCPP_HIDE_FROM_ABI constexpr __iterator(const char* const* __enc_d) noexcept : __data_(__enc_d) {}

      const char* const* __data_;
    }; // __iterator

    _LIBCPP_HIDE_FROM_ABI constexpr __iterator begin() const { return __iterator{__view_data_->__aliases_begin()}; }
    _LIBCPP_HIDE_FROM_ABI constexpr __iterator end() const { return __iterator{__view_data_->__aliases_end()}; }

  private:
    friend struct __te_impl;

    _LIBCPP_HIDE_FROM_ABI constexpr __aliases_view(const __te_data* __d) : __view_data_(__d) {}
    const __te_data* __view_data_;
  }; // __aliases_view

  [[__nodiscard__]] _LIBCPP_HIDE_FROM_ABI constexpr __aliases_view __aliases() const {
    return __aliases_view(&__entries[__encoding_idx_]);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __te_impl& __a, const __te_impl& __b) noexcept {
    return __a.__mib() == __id::other && __b.__mib() == __id::other
             ? __comp_name(__a.__name_, __b.__name_)
             : __a.__mib() == __b.__mib();
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __te_impl& __encoding, __id __i) noexcept {
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
  unsigned short __encoding_idx_{1u};
  char __name_[__max_name_length_ + 1]{};

  // Structuring the text encoding data and the table in this manner has a few benefits:
  // Smaller footprint: We only need 6 bytes for the information we need.
  // Smaller total table: We only need one entry per Mib.
  // Easier aliases_view: We know the begin, and end of each range of aliases, which makes iterator
  // implementation simple.

  static constexpr __te_data __entries[] = {
      {0, 1, 0},      {1, 2, 0},      {2, 3, 11},     {13, 4, 9},     {22, 5, 7},     {29, 6, 7},     {36, 7, 7},
      {43, 8, 6},     {49, 9, 8},     {57, 10, 9},    {66, 11, 6},    {72, 12, 7},    {79, 13, 6},    {85, 14, 3},
      {88, 15, 3},    {91, 16, 2},    {93, 17, 3},    {96, 18, 3},    {99, 19, 2},    {101, 20, 6},   {107, 21, 5},
      {112, 22, 4},   {116, 23, 4},   {120, 24, 5},   {125, 25, 6},   {131, 26, 5},   {136, 27, 2},   {138, 28, 3},
      {141, 29, 2},   {143, 30, 4},   {147, 31, 3},   {150, 32, 3},   {153, 35, 7},   {160, 36, 6},   {166, 37, 2},
      {168, 38, 2},   {170, 39, 2},   {172, 40, 2},   {174, 41, 6},   {180, 42, 5},   {185, 43, 4},   {189, 44, 3},
      {192, 45, 3},   {195, 46, 4},   {199, 47, 3},   {202, 48, 3},   {205, 49, 3},   {208, 50, 3},   {211, 51, 3},
      {214, 52, 3},   {217, 53, 3},   {220, 54, 4},   {224, 55, 3},   {227, 56, 5},   {232, 57, 4},   {236, 58, 5},
      {241, 59, 3},   {244, 60, 4},   {248, 61, 4},   {252, 62, 5},   {257, 63, 5},   {262, 64, 3},   {265, 65, 5},
      {270, 66, 2},   {272, 67, 4},   {276, 68, 5},   {281, 69, 4},   {285, 70, 4},   {289, 71, 4},   {293, 72, 3},
      {296, 73, 4},   {300, 74, 5},   {305, 75, 3},   {308, 76, 4},   {312, 77, 4},   {316, 78, 7},   {323, 79, 6},
      {329, 80, 3},   {332, 81, 3},   {335, 82, 3},   {338, 83, 3},   {341, 84, 3},   {344, 85, 3},   {347, 86, 3},
      {350, 87, 6},   {356, 88, 3},   {359, 89, 4},   {363, 90, 4},   {367, 91, 4},   {371, 92, 5},   {376, 93, 3},
      {379, 94, 4},   {383, 95, 4},   {387, 96, 3},   {390, 97, 4},   {394, 98, 4},   {398, 99, 5},   {403, 100, 2},
      {405, 101, 2},  {407, 102, 3},  {410, 103, 2},  {412, 104, 2},  {414, 105, 2},  {416, 106, 2},  {418, 109, 2},
      {420, 110, 8},  {428, 111, 4},  {432, 112, 7},  {439, 113, 5},  {444, 114, 2},  {446, 115, 2},  {448, 116, 2},
      {450, 117, 2},  {452, 118, 4},  {456, 119, 4},  {460, 1000, 2}, {462, 1001, 2}, {464, 1002, 2}, {466, 1003, 3},
      {469, 1004, 2}, {471, 1005, 2}, {473, 1006, 2}, {475, 1007, 2}, {477, 1008, 2}, {479, 1009, 2}, {481, 1010, 2},
      {483, 1011, 2}, {485, 1012, 2}, {487, 1013, 2}, {489, 1014, 2}, {491, 1015, 2}, {493, 1016, 3}, {496, 1017, 2},
      {498, 1018, 2}, {500, 1019, 2}, {502, 1020, 3}, {505, 1021, 2}, {507, 2000, 2}, {509, 2001, 2}, {511, 2002, 2},
      {513, 2003, 2}, {515, 2004, 4}, {519, 2005, 2}, {521, 2006, 2}, {523, 2007, 2}, {525, 2008, 3}, {528, 2009, 4},
      {532, 2010, 4}, {536, 2011, 4}, {540, 2012, 2}, {542, 2013, 4}, {546, 2014, 2}, {548, 2015, 2}, {550, 2016, 2},
      {552, 2017, 2}, {554, 2018, 2}, {556, 2019, 2}, {558, 2020, 2}, {560, 2021, 2}, {562, 2022, 2}, {564, 2023, 2},
      {566, 2024, 2}, {568, 2025, 2}, {570, 2026, 2}, {572, 2027, 3}, {575, 2028, 7}, {582, 2029, 4}, {586, 2030, 3},
      {589, 2031, 4}, {593, 2032, 4}, {597, 2033, 4}, {601, 2034, 5}, {606, 2035, 4}, {610, 2036, 4}, {614, 2037, 4},
      {618, 2038, 4}, {622, 2039, 4}, {626, 2040, 4}, {630, 2041, 4}, {634, 2042, 4}, {638, 2043, 4}, {642, 2044, 5},
      {647, 2045, 4}, {651, 2046, 4}, {655, 2047, 4}, {659, 2048, 4}, {663, 2049, 5}, {668, 2050, 4}, {672, 2051, 3},
      {675, 2052, 4}, {679, 2053, 4}, {683, 2054, 5}, {688, 2055, 5}, {693, 2056, 4}, {697, 2057, 4}, {701, 2058, 3},
      {704, 2059, 3}, {707, 2060, 4}, {711, 2061, 4}, {715, 2062, 4}, {719, 2063, 3}, {722, 2064, 2}, {724, 2065, 2},
      {726, 2066, 2}, {728, 2067, 2}, {730, 2068, 2}, {732, 2069, 2}, {734, 2070, 2}, {736, 2071, 2}, {738, 2072, 2},
      {740, 2073, 2}, {742, 2074, 2}, {744, 2075, 2}, {746, 2076, 2}, {748, 2077, 2}, {750, 2078, 2}, {752, 2079, 2},
      {754, 2080, 2}, {756, 2081, 2}, {758, 2082, 2}, {760, 2083, 2}, {762, 2084, 2}, {764, 2085, 1}, {765, 2086, 4},
      {769, 2087, 3}, {772, 2088, 2}, {774, 2089, 5}, {779, 2090, 5}, {784, 2091, 5}, {789, 2092, 5}, {794, 2093, 6},
      {800, 2094, 6}, {806, 2095, 5}, {811, 2096, 5}, {816, 2097, 5}, {821, 2098, 5}, {826, 2099, 5}, {831, 2100, 5},
      {836, 2101, 2}, {838, 2102, 3}, {841, 2103, 5}, {846, 2104, 5}, {851, 2105, 2}, {853, 2106, 2}, {855, 2107, 2},
      {857, 2108, 2}, {859, 2109, 2}, {861, 2250, 2}, {863, 2251, 2}, {865, 2252, 2}, {867, 2253, 2}, {869, 2254, 2},
      {871, 2255, 2}, {873, 2256, 2}, {875, 2257, 2}, {877, 2258, 2}, {879, 2259, 3}, {882, 2260, 2}};

  static constexpr char const* __aliases_table[] = {
      "",
      "",
      "US-ASCII",
      "iso-ir-6",
      "ANSI_X3.4-1968",
      "ANSI_X3.4-1986",
      "ISO_646.irv:1991",
      "ISO646-US",
      "us",
      "IBM367",
      "cp367",
      "csASCII",
      "ASCII",
      "ISO_8859-1:1987",
      "iso-ir-100",
      "ISO_8859-1",
      "ISO-8859-1",
      "latin1",
      "l1",
      "IBM819",
      "CP819",
      "csISOLatin1",
      "ISO_8859-2:1987",
      "iso-ir-101",
      "ISO_8859-2",
      "ISO-8859-2",
      "latin2",
      "l2",
      "csISOLatin2",
      "ISO_8859-3:1988",
      "iso-ir-109",
      "ISO_8859-3",
      "ISO-8859-3",
      "latin3",
      "l3",
      "csISOLatin3",
      "ISO_8859-4:1988",
      "iso-ir-110",
      "ISO_8859-4",
      "ISO-8859-4",
      "latin4",
      "l4",
      "csISOLatin4",
      "ISO_8859-5:1988",
      "iso-ir-144",
      "ISO_8859-5",
      "ISO-8859-5",
      "cyrillic",
      "csISOLatinCyrillic",
      "ISO_8859-6:1987",
      "iso-ir-127",
      "ISO_8859-6",
      "ISO-8859-6",
      "ECMA-114",
      "ASMO-708",
      "arabic",
      "csISOLatinArabic",
      "ISO_8859-7:1987",
      "iso-ir-126",
      "ISO_8859-7",
      "ISO-8859-7",
      "ELOT_928",
      "ECMA-118",
      "greek",
      "greek8",
      "csISOLatinGreek",
      "ISO_8859-8:1988",
      "iso-ir-138",
      "ISO_8859-8",
      "ISO-8859-8",
      "hebrew",
      "csISOLatinHebrew",
      "ISO_8859-9:1989",
      "iso-ir-148",
      "ISO_8859-9",
      "ISO-8859-9",
      "latin5",
      "l5",
      "csISOLatin5",
      "ISO-8859-10",
      "iso-ir-157",
      "l6",
      "ISO_8859-10:1992",
      "csISOLatin6",
      "latin6",
      "ISO_6937-2-add",
      "iso-ir-142",
      "csISOTextComm",
      "JIS_X0201",
      "X0201",
      "csHalfWidthKatakana",
      "JIS_Encoding",
      "csJISEncoding",
      "Shift_JIS",
      "MS_Kanji",
      "csShiftJIS",
      "Extended_UNIX_Code_Packed_Format_for_Japanese",
      "csEUCPkdFmtJapanese",
      "EUC-JP",
      "Extended_UNIX_Code_Fixed_Width_for_Japanese",
      "csEUCFixWidJapanese",
      "BS_4730",
      "iso-ir-4",
      "ISO646-GB",
      "gb",
      "uk",
      "csISO4UnitedKingdom",
      "SEN_850200_C",
      "iso-ir-11",
      "ISO646-SE2",
      "se2",
      "csISO11SwedishForNames",
      "IT",
      "iso-ir-15",
      "ISO646-IT",
      "csISO15Italian",
      "ES",
      "iso-ir-17",
      "ISO646-ES",
      "csISO17Spanish",
      "DIN_66003",
      "iso-ir-21",
      "de",
      "ISO646-DE",
      "csISO21German",
      "NS_4551-1",
      "iso-ir-60",
      "ISO646-NO",
      "no",
      "csISO60DanishNorwegian",
      "csISO60Norwegian1",
      "NF_Z_62-010",
      "iso-ir-69",
      "ISO646-FR",
      "fr",
      "csISO69French",
      "ISO-10646-UTF-1",
      "csISO10646UTF1",
      "ISO_646.basic:1983",
      "ref",
      "csISO646basic1983",
      "INVARIANT",
      "csINVARIANT",
      "ISO_646.irv:1983",
      "iso-ir-2",
      "irv",
      "csISO2IntlRefVersion",
      "NATS-SEFI",
      "iso-ir-8-1",
      "csNATSSEFI",
      "NATS-SEFI-ADD",
      "iso-ir-8-2",
      "csNATSSEFIADD",
      "SEN_850200_B",
      "iso-ir-10",
      "FI",
      "ISO646-FI",
      "ISO646-SE",
      "se",
      "csISO10Swedish",
      "KS_C_5601-1987",
      "iso-ir-149",
      "KS_C_5601-1989",
      "KSC_5601",
      "korean",
      "csKSC56011987",
      "ISO-2022-KR",
      "csISO2022KR",
      "EUC-KR",
      "csEUCKR",
      "ISO-2022-JP",
      "csISO2022JP",
      "ISO-2022-JP-2",
      "csISO2022JP2",
      "JIS_C6220-1969-jp",
      "JIS_C6220-1969",
      "iso-ir-13",
      "katakana",
      "x0201-7",
      "csISO13JISC6220jp",
      "JIS_C6220-1969-ro",
      "iso-ir-14",
      "jp",
      "ISO646-JP",
      "csISO14JISC6220ro",
      "PT",
      "iso-ir-16",
      "ISO646-PT",
      "csISO16Portuguese",
      "greek7-old",
      "iso-ir-18",
      "csISO18Greek7Old",
      "latin-greek",
      "iso-ir-19",
      "csISO19LatinGreek",
      "NF_Z_62-010_(1973)",
      "iso-ir-25",
      "ISO646-FR1",
      "csISO25French",
      "Latin-greek-1",
      "iso-ir-27",
      "csISO27LatinGreek1",
      "ISO_5427",
      "iso-ir-37",
      "csISO5427Cyrillic",
      "JIS_C6226-1978",
      "iso-ir-42",
      "csISO42JISC62261978",
      "BS_viewdata",
      "iso-ir-47",
      "csISO47BSViewdata",
      "INIS",
      "iso-ir-49",
      "csISO49INIS",
      "INIS-8",
      "iso-ir-50",
      "csISO50INIS8",
      "INIS-cyrillic",
      "iso-ir-51",
      "csISO51INISCyrillic",
      "ISO_5427:1981",
      "iso-ir-54",
      "ISO5427Cyrillic1981",
      "csISO54271981",
      "ISO_5428:1980",
      "iso-ir-55",
      "csISO5428Greek",
      "GB_1988-80",
      "iso-ir-57",
      "cn",
      "ISO646-CN",
      "csISO57GB1988",
      "GB_2312-80",
      "iso-ir-58",
      "chinese",
      "csISO58GB231280",
      "NS_4551-2",
      "ISO646-NO2",
      "iso-ir-61",
      "no2",
      "csISO61Norwegian2",
      "videotex-suppl",
      "iso-ir-70",
      "csISO70VideotexSupp1",
      "PT2",
      "iso-ir-84",
      "ISO646-PT2",
      "csISO84Portuguese2",
      "ES2",
      "iso-ir-85",
      "ISO646-ES2",
      "csISO85Spanish2",
      "MSZ_7795.3",
      "iso-ir-86",
      "ISO646-HU",
      "hu",
      "csISO86Hungarian",
      "JIS_C6226-1983",
      "iso-ir-87",
      "x0208",
      "JIS_X0208-1983",
      "csISO87JISX0208",
      "greek7",
      "iso-ir-88",
      "csISO88Greek7",
      "ASMO_449",
      "ISO_9036",
      "arabic7",
      "iso-ir-89",
      "csISO89ASMO449",
      "iso-ir-90",
      "csISO90",
      "JIS_C6229-1984-a",
      "iso-ir-91",
      "jp-ocr-a",
      "csISO91JISC62291984a",
      "JIS_C6229-1984-b",
      "iso-ir-92",
      "ISO646-JP-OCR-B",
      "jp-ocr-b",
      "csISO92JISC62991984b",
      "JIS_C6229-1984-b-add",
      "iso-ir-93",
      "jp-ocr-b-add",
      "csISO93JIS62291984badd",
      "JIS_C6229-1984-hand",
      "iso-ir-94",
      "jp-ocr-hand",
      "csISO94JIS62291984hand",
      "JIS_C6229-1984-hand-add",
      "iso-ir-95",
      "jp-ocr-hand-add",
      "csISO95JIS62291984handadd",
      "JIS_C6229-1984-kana",
      "iso-ir-96",
      "csISO96JISC62291984kana",
      "ISO_2033-1983",
      "iso-ir-98",
      "e13b",
      "csISO2033",
      "ANSI_X3.110-1983",
      "iso-ir-99",
      "CSA_T500-1983",
      "NAPLPS",
      "csISO99NAPLPS",
      "T.61-7bit",
      "iso-ir-102",
      "csISO102T617bit",
      "T.61-8bit",
      "T.61",
      "iso-ir-103",
      "csISO103T618bit",
      "ECMA-cyrillic",
      "iso-ir-111",
      "KOI8-E",
      "csISO111ECMACyrillic",
      "CSA_Z243.4-1985-1",
      "iso-ir-121",
      "ISO646-CA",
      "csa7-1",
      "csa71",
      "ca",
      "csISO121Canadian1",
      "CSA_Z243.4-1985-2",
      "iso-ir-122",
      "ISO646-CA2",
      "csa7-2",
      "csa72",
      "csISO122Canadian2",
      "CSA_Z243.4-1985-gr",
      "iso-ir-123",
      "csISO123CSAZ24341985gr",
      "ISO_8859-6-E",
      "csISO88596E",
      "ISO-8859-6-E",
      "ISO_8859-6-I",
      "csISO88596I",
      "ISO-8859-6-I",
      "T.101-G2",
      "iso-ir-128",
      "csISO128T101G2",
      "ISO_8859-8-E",
      "csISO88598E",
      "ISO-8859-8-E",
      "ISO_8859-8-I",
      "csISO88598I",
      "ISO-8859-8-I",
      "CSN_369103",
      "iso-ir-139",
      "csISO139CSN369103",
      "JUS_I.B1.002",
      "iso-ir-141",
      "ISO646-YU",
      "js",
      "yu",
      "csISO141JUSIB1002",
      "IEC_P27-1",
      "iso-ir-143",
      "csISO143IECP271",
      "JUS_I.B1.003-serb",
      "iso-ir-146",
      "serbian",
      "csISO146Serbian",
      "JUS_I.B1.003-mac",
      "macedonian",
      "iso-ir-147",
      "csISO147Macedonian",
      "greek-ccitt",
      "iso-ir-150",
      "csISO150",
      "csISO150GreekCCITT",
      "NC_NC00-10:81",
      "cuba",
      "iso-ir-151",
      "ISO646-CU",
      "csISO151Cuba",
      "ISO_6937-2-25",
      "iso-ir-152",
      "csISO6937Add",
      "GOST_19768-74",
      "ST_SEV_358-88",
      "iso-ir-153",
      "csISO153GOST1976874",
      "ISO_8859-supp",
      "iso-ir-154",
      "latin1-2-5",
      "csISO8859Supp",
      "ISO_10367-box",
      "iso-ir-155",
      "csISO10367Box",
      "latin-lap",
      "lap",
      "iso-ir-158",
      "csISO158Lap",
      "JIS_X0212-1990",
      "x0212",
      "iso-ir-159",
      "csISO159JISX02121990",
      "DS_2089",
      "DS2089",
      "ISO646-DK",
      "dk",
      "csISO646Danish",
      "us-dk",
      "csUSDK",
      "dk-us",
      "csDKUS",
      "KSC5636",
      "ISO646-KR",
      "csKSC5636",
      "UNICODE-1-1-UTF-7",
      "csUnicode11UTF7",
      "ISO-2022-CN",
      "csISO2022CN",
      "ISO-2022-CN-EXT",
      "csISO2022CNEXT",
      "UTF-8",
      "csUTF8",
      "ISO-8859-13",
      "csISO885913",
      "ISO-8859-14",
      "iso-ir-199",
      "ISO_8859-14:1998",
      "ISO_8859-14",
      "latin8",
      "iso-celtic",
      "l8",
      "csISO885914",
      "ISO-8859-15",
      "ISO_8859-15",
      "Latin-9",
      "csISO885915",
      "ISO-8859-16",
      "iso-ir-226",
      "ISO_8859-16:2001",
      "ISO_8859-16",
      "latin10",
      "l10",
      "csISO885916",
      "GBK",
      "CP936",
      "MS936",
      "windows-936",
      "csGBK",
      "GB18030",
      "csGB18030",
      "OSD_EBCDIC_DF04_15",
      "csOSDEBCDICDF0415",
      "OSD_EBCDIC_DF03_IRV",
      "csOSDEBCDICDF03IRV",
      "OSD_EBCDIC_DF04_1",
      "csOSDEBCDICDF041",
      "ISO-11548-1",
      "ISO_11548-1",
      "ISO_TR_11548-1",
      "csISO115481",
      "KZ-1048",
      "STRK1048-2002",
      "RK1048",
      "csKZ1048",
      "ISO-10646-UCS-2",
      "csUnicode",
      "ISO-10646-UCS-4",
      "csUCS4",
      "ISO-10646-UCS-Basic",
      "csUnicodeASCII",
      "ISO-10646-Unicode-Latin1",
      "csUnicodeLatin1",
      "ISO-10646",
      "ISO-10646-J-1",
      "csUnicodeJapanese",
      "ISO-Unicode-IBM-1261",
      "csUnicodeIBM1261",
      "ISO-Unicode-IBM-1268",
      "csUnicodeIBM1268",
      "ISO-Unicode-IBM-1276",
      "csUnicodeIBM1276",
      "ISO-Unicode-IBM-1264",
      "csUnicodeIBM1264",
      "ISO-Unicode-IBM-1265",
      "csUnicodeIBM1265",
      "UNICODE-1-1",
      "csUnicode11",
      "SCSU",
      "csSCSU",
      "UTF-7",
      "csUTF7",
      "UTF-16BE",
      "csUTF16BE",
      "UTF-16LE",
      "csUTF16LE",
      "UTF-16",
      "csUTF16",
      "CESU-8",
      "csCESU8",
      "csCESU-8",
      "UTF-32",
      "csUTF32",
      "UTF-32BE",
      "csUTF32BE",
      "UTF-32LE",
      "csUTF32LE",
      "BOCU-1",
      "csBOCU1",
      "csBOCU-1",
      "UTF-7-IMAP",
      "csUTF7IMAP",
      "ISO-8859-1-Windows-3.0-Latin-1",
      "csWindows30Latin1",
      "ISO-8859-1-Windows-3.1-Latin-1",
      "csWindows31Latin1",
      "ISO-8859-2-Windows-Latin-2",
      "csWindows31Latin2",
      "ISO-8859-9-Windows-Latin-5",
      "csWindows31Latin5",
      "hp-roman8",
      "roman8",
      "r8",
      "csHPRoman8",
      "Adobe-Standard-Encoding",
      "csAdobeStandardEncoding",
      "Ventura-US",
      "csVenturaUS",
      "Ventura-International",
      "csVenturaInternational",
      "DEC-MCS",
      "dec",
      "csDECMCS",
      "IBM850",
      "cp850",
      "850",
      "csPC850Multilingual",
      "IBM852",
      "cp852",
      "852",
      "csPCp852",
      "IBM437",
      "cp437",
      "437",
      "csPC8CodePage437",
      "PC8-Danish-Norwegian",
      "csPC8DanishNorwegian",
      "IBM862",
      "cp862",
      "862",
      "csPC862LatinHebrew",
      "PC8-Turkish",
      "csPC8Turkish",
      "IBM-Symbols",
      "csIBMSymbols",
      "IBM-Thai",
      "csIBMThai",
      "HP-Legal",
      "csHPLegal",
      "HP-Pi-font",
      "csHPPiFont",
      "HP-Math8",
      "csHPMath8",
      "Adobe-Symbol-Encoding",
      "csHPPSMath",
      "HP-DeskTop",
      "csHPDesktop",
      "Ventura-Math",
      "csVenturaMath",
      "Microsoft-Publishing",
      "csMicrosoftPublishing",
      "Windows-31J",
      "csWindows31J",
      "GB2312",
      "csGB2312",
      "Big5",
      "csBig5",
      "macintosh",
      "mac",
      "csMacintosh",
      "IBM037",
      "cp037",
      "ebcdic-cp-us",
      "ebcdic-cp-ca",
      "ebcdic-cp-wt",
      "ebcdic-cp-nl",
      "csIBM037",
      "IBM038",
      "EBCDIC-INT",
      "cp038",
      "csIBM038",
      "IBM273",
      "CP273",
      "csIBM273",
      "IBM274",
      "EBCDIC-BE",
      "CP274",
      "csIBM274",
      "IBM275",
      "EBCDIC-BR",
      "cp275",
      "csIBM275",
      "IBM277",
      "EBCDIC-CP-DK",
      "EBCDIC-CP-NO",
      "csIBM277",
      "IBM278",
      "CP278",
      "ebcdic-cp-fi",
      "ebcdic-cp-se",
      "csIBM278",
      "IBM280",
      "CP280",
      "ebcdic-cp-it",
      "csIBM280",
      "IBM281",
      "EBCDIC-JP-E",
      "cp281",
      "csIBM281",
      "IBM284",
      "CP284",
      "ebcdic-cp-es",
      "csIBM284",
      "IBM285",
      "CP285",
      "ebcdic-cp-gb",
      "csIBM285",
      "IBM290",
      "cp290",
      "EBCDIC-JP-kana",
      "csIBM290",
      "IBM297",
      "cp297",
      "ebcdic-cp-fr",
      "csIBM297",
      "IBM420",
      "cp420",
      "ebcdic-cp-ar1",
      "csIBM420",
      "IBM423",
      "cp423",
      "ebcdic-cp-gr",
      "csIBM423",
      "IBM424",
      "cp424",
      "ebcdic-cp-he",
      "csIBM424",
      "IBM500",
      "CP500",
      "ebcdic-cp-be",
      "ebcdic-cp-ch",
      "csIBM500",
      "IBM851",
      "cp851",
      "851",
      "csIBM851",
      "IBM855",
      "cp855",
      "855",
      "csIBM855",
      "IBM857",
      "cp857",
      "857",
      "csIBM857",
      "IBM860",
      "cp860",
      "860",
      "csIBM860",
      "IBM861",
      "cp861",
      "861",
      "cp-is",
      "csIBM861",
      "IBM863",
      "cp863",
      "863",
      "csIBM863",
      "IBM864",
      "cp864",
      "csIBM864",
      "IBM865",
      "cp865",
      "865",
      "csIBM865",
      "IBM868",
      "CP868",
      "cp-ar",
      "csIBM868",
      "IBM869",
      "cp869",
      "869",
      "cp-gr",
      "csIBM869",
      "IBM870",
      "CP870",
      "ebcdic-cp-roece",
      "ebcdic-cp-yu",
      "csIBM870",
      "IBM871",
      "CP871",
      "ebcdic-cp-is",
      "csIBM871",
      "IBM880",
      "cp880",
      "EBCDIC-Cyrillic",
      "csIBM880",
      "IBM891",
      "cp891",
      "csIBM891",
      "IBM903",
      "cp903",
      "csIBM903",
      "IBM904",
      "cp904",
      "904",
      "csIBBM904",
      "IBM905",
      "CP905",
      "ebcdic-cp-tr",
      "csIBM905",
      "IBM918",
      "CP918",
      "ebcdic-cp-ar2",
      "csIBM918",
      "IBM1026",
      "CP1026",
      "csIBM1026",
      "EBCDIC-AT-DE",
      "csIBMEBCDICATDE",
      "EBCDIC-AT-DE-A",
      "csEBCDICATDEA",
      "EBCDIC-CA-FR",
      "csEBCDICCAFR",
      "EBCDIC-DK-NO",
      "csEBCDICDKNO",
      "EBCDIC-DK-NO-A",
      "csEBCDICDKNOA",
      "EBCDIC-FI-SE",
      "csEBCDICFISE",
      "EBCDIC-FI-SE-A",
      "csEBCDICFISEA",
      "EBCDIC-FR",
      "csEBCDICFR",
      "EBCDIC-IT",
      "csEBCDICIT",
      "EBCDIC-PT",
      "csEBCDICPT",
      "EBCDIC-ES",
      "csEBCDICES",
      "EBCDIC-ES-A",
      "csEBCDICESA",
      "EBCDIC-ES-S",
      "csEBCDICESS",
      "EBCDIC-UK",
      "csEBCDICUK",
      "EBCDIC-US",
      "csEBCDICUS",
      "UNKNOWN-8BIT",
      "csUnknown8BiT",
      "MNEMONIC",
      "csMnemonic",
      "MNEM",
      "csMnem",
      "VISCII",
      "csVISCII",
      "VIQR",
      "csVIQR",
      "KOI8-R",
      "csKOI8R",
      "HZ-GB-2312",
      "IBM866",
      "cp866",
      "866",
      "csIBM866",
      "IBM775",
      "cp775",
      "csPC775Baltic",
      "KOI8-U",
      "csKOI8U",
      "IBM00858",
      "CCSID00858",
      "CP00858",
      "PC-Multilingual-850+euro",
      "csIBM00858",
      "IBM00924",
      "CCSID00924",
      "CP00924",
      "ebcdic-Latin9--euro",
      "csIBM00924",
      "IBM01140",
      "CCSID01140",
      "CP01140",
      "ebcdic-us-37+euro",
      "csIBM01140",
      "IBM01141",
      "CCSID01141",
      "CP01141",
      "ebcdic-de-273+euro",
      "csIBM01141",
      "IBM01142",
      "CCSID01142",
      "CP01142",
      "ebcdic-dk-277+euro",
      "ebcdic-no-277+euro",
      "csIBM01142",
      "IBM01143",
      "CCSID01143",
      "CP01143",
      "ebcdic-fi-278+euro",
      "ebcdic-se-278+euro",
      "csIBM01143",
      "IBM01144",
      "CCSID01144",
      "CP01144",
      "ebcdic-it-280+euro",
      "csIBM01144",
      "IBM01145",
      "CCSID01145",
      "CP01145",
      "ebcdic-es-284+euro",
      "csIBM01145",
      "IBM01146",
      "CCSID01146",
      "CP01146",
      "ebcdic-gb-285+euro",
      "csIBM01146",
      "IBM01147",
      "CCSID01147",
      "CP01147",
      "ebcdic-fr-297+euro",
      "csIBM01147",
      "IBM01148",
      "CCSID01148",
      "CP01148",
      "ebcdic-international-500+euro",
      "csIBM01148",
      "IBM01149",
      "CCSID01149",
      "CP01149",
      "ebcdic-is-871+euro",
      "csIBM01149",
      "Big5-HKSCS",
      "csBig5HKSCS",
      "IBM1047",
      "IBM-1047",
      "csIBM1047",
      "PTCP154",
      "csPTCP154",
      "PT154",
      "CP154",
      "Cyrillic-Asian",
      "Amiga-1251",
      "Ami1251",
      "Amiga1251",
      "Ami-1251",
      "csAmiga1251",
      "KOI7-switched",
      "csKOI7switched",
      "BRF",
      "csBRF",
      "TSCII",
      "csTSCII",
      "CP51932",
      "csCP51932",
      "windows-874",
      "cswindows874",
      "windows-1250",
      "cswindows1250",
      "windows-1251",
      "cswindows1251",
      "windows-1252",
      "cswindows1252",
      "windows-1253",
      "cswindows1253",
      "windows-1254",
      "cswindows1254",
      "windows-1255",
      "cswindows1255",
      "windows-1256",
      "cswindows1256",
      "windows-1257",
      "cswindows1257",
      "windows-1258",
      "cswindows1258",
      "TIS-620",
      "csTIS620",
      "ISO-8859-11",
      "CP50220",
      "csCP50220",
  };
};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_POP_MACROS

#endif // _LIBCPP__TEXT_ENCODING_TE_IMPL_H
