/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf -acCgopt -k'1,2,5,9,$' -L ANSI-C -N locfile_hash locfile-kw.gperf  */

#if !((' ' == 32) && ('!' == 33) && ('"' == 34) && ('#' == 35) \
      && ('%' == 37) && ('&' == 38) && ('\'' == 39) && ('(' == 40) \
      && (')' == 41) && ('*' == 42) && ('+' == 43) && (',' == 44) \
      && ('-' == 45) && ('.' == 46) && ('/' == 47) && ('0' == 48) \
      && ('1' == 49) && ('2' == 50) && ('3' == 51) && ('4' == 52) \
      && ('5' == 53) && ('6' == 54) && ('7' == 55) && ('8' == 56) \
      && ('9' == 57) && (':' == 58) && (';' == 59) && ('<' == 60) \
      && ('=' == 61) && ('>' == 62) && ('?' == 63) && ('A' == 65) \
      && ('B' == 66) && ('C' == 67) && ('D' == 68) && ('E' == 69) \
      && ('F' == 70) && ('G' == 71) && ('H' == 72) && ('I' == 73) \
      && ('J' == 74) && ('K' == 75) && ('L' == 76) && ('M' == 77) \
      && ('N' == 78) && ('O' == 79) && ('P' == 80) && ('Q' == 81) \
      && ('R' == 82) && ('S' == 83) && ('T' == 84) && ('U' == 85) \
      && ('V' == 86) && ('W' == 87) && ('X' == 88) && ('Y' == 89) \
      && ('Z' == 90) && ('[' == 91) && ('\\' == 92) && (']' == 93) \
      && ('^' == 94) && ('_' == 95) && ('a' == 97) && ('b' == 98) \
      && ('c' == 99) && ('d' == 100) && ('e' == 101) && ('f' == 102) \
      && ('g' == 103) && ('h' == 104) && ('i' == 105) && ('j' == 106) \
      && ('k' == 107) && ('l' == 108) && ('m' == 109) && ('n' == 110) \
      && ('o' == 111) && ('p' == 112) && ('q' == 113) && ('r' == 114) \
      && ('s' == 115) && ('t' == 116) && ('u' == 117) && ('v' == 118) \
      && ('w' == 119) && ('x' == 120) && ('y' == 121) && ('z' == 122) \
      && ('{' == 123) && ('|' == 124) && ('}' == 125) && ('~' == 126))
/* The character set is not based on ISO-646.  */
#error "gperf generated tables don't work with this execution character set. Please report a bug to <bug-gperf@gnu.org>."
#endif

#line 1 "locfile-kw.gperf"

/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1996.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published
   by the Free Software Foundation; version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <string.h>

#include "locfile-token.h"
#line 24 "locfile-kw.gperf"
struct keyword_t ;

#define TOTAL_KEYWORDS 178
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 22
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 630
/* maximum key range = 628, duplicates = 0 */

#ifdef __GNUC__
__inline
#else
#ifdef __cplusplus
inline
#endif
#endif
static unsigned int
hash (register const char *str, register size_t len)
{
  static const unsigned short asso_values[] =
    {
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
        5,   0, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631,   5, 631,   0,   0,   0,
        0,   0,  10,   0, 631, 631,   0, 631,   0,   5,
      631, 631,   0,   0,   0,  10, 631, 631, 631,   0,
      631, 631, 631, 631, 631,   0, 631, 145,  80,  25,
       15,   0, 180, 105,  10,  35, 631,  50,  80, 160,
        5, 130,  40,  45,   5,   0,  10,  35,  40,  35,
        5,  10,   0, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631, 631, 631, 631, 631,
      631, 631, 631, 631, 631, 631
    };
  register unsigned int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[8]];
      /*FALLTHROUGH*/
      case 8:
      case 7:
      case 6:
      case 5:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
      case 3:
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

const struct keyword_t *
locfile_hash (register const char *str, register size_t len)
{
  static const struct keyword_t wordlist[] =
    {
      {""}, {""}, {""},
#line 31 "locfile-kw.gperf"
      {"END",                    tok_end,                    0},
      {""}, {""},
#line 70 "locfile-kw.gperf"
      {"IGNORE",                 tok_ignore,                 0},
#line 129 "locfile-kw.gperf"
      {"LC_TIME",                tok_lc_time,                0},
#line 30 "locfile-kw.gperf"
      {"LC_CTYPE",               tok_lc_ctype,               0},
      {""},
#line 168 "locfile-kw.gperf"
      {"LC_ADDRESS",             tok_lc_address,             0},
#line 153 "locfile-kw.gperf"
      {"LC_MESSAGES",            tok_lc_messages,            0},
#line 161 "locfile-kw.gperf"
      {"LC_NAME",                tok_lc_name,                0},
#line 158 "locfile-kw.gperf"
      {"LC_PAPER",               tok_lc_paper,               0},
#line 186 "locfile-kw.gperf"
      {"LC_MEASUREMENT",         tok_lc_measurement,         0},
#line 56 "locfile-kw.gperf"
      {"LC_COLLATE",             tok_lc_collate,             0},
      {""},
#line 188 "locfile-kw.gperf"
      {"LC_IDENTIFICATION",      tok_lc_identification,      0},
#line 201 "locfile-kw.gperf"
      {"revision",               tok_revision,               0},
#line 69 "locfile-kw.gperf"
      {"UNDEFINED",              tok_undefined,              0},
#line 125 "locfile-kw.gperf"
      {"LC_NUMERIC",             tok_lc_numeric,             0},
#line 82 "locfile-kw.gperf"
      {"LC_MONETARY",            tok_lc_monetary,            0},
#line 181 "locfile-kw.gperf"
      {"LC_TELEPHONE",           tok_lc_telephone,           0},
      {""}, {""}, {""},
#line 75 "locfile-kw.gperf"
      {"define",                 tok_define,                 0},
#line 154 "locfile-kw.gperf"
      {"yesexpr",                tok_yesexpr,                0},
#line 141 "locfile-kw.gperf"
      {"era_year",               tok_era_year,               0},
      {""},
#line 54 "locfile-kw.gperf"
      {"translit_ignore",        tok_translit_ignore,        0},
#line 156 "locfile-kw.gperf"
      {"yesstr",                 tok_yesstr,                 0},
      {""},
#line 89 "locfile-kw.gperf"
      {"negative_sign",          tok_negative_sign,          0},
      {""},
#line 137 "locfile-kw.gperf"
      {"t_fmt",                  tok_t_fmt,                  0},
#line 159 "locfile-kw.gperf"
      {"height",                 tok_height,                 0},
      {""}, {""},
#line 52 "locfile-kw.gperf"
      {"translit_start",         tok_translit_start,         0},
#line 136 "locfile-kw.gperf"
      {"d_fmt",                  tok_d_fmt,                  0},
      {""},
#line 53 "locfile-kw.gperf"
      {"translit_end",           tok_translit_end,           0},
#line 94 "locfile-kw.gperf"
      {"n_cs_precedes",          tok_n_cs_precedes,          0},
#line 144 "locfile-kw.gperf"
      {"era_t_fmt",              tok_era_t_fmt,              0},
#line 39 "locfile-kw.gperf"
      {"space",                  tok_space,                  0},
#line 72 "locfile-kw.gperf"
      {"reorder-end",            tok_reorder_end,            0},
#line 73 "locfile-kw.gperf"
      {"reorder-sections-after", tok_reorder_sections_after, 0},
      {""},
#line 142 "locfile-kw.gperf"
      {"era_d_fmt",              tok_era_d_fmt,              0},
#line 189 "locfile-kw.gperf"
      {"title",                  tok_title,                  0},
      {""}, {""},
#line 149 "locfile-kw.gperf"
      {"timezone",               tok_timezone,               0},
      {""},
#line 74 "locfile-kw.gperf"
      {"reorder-sections-end",   tok_reorder_sections_end,   0},
      {""}, {""}, {""},
#line 95 "locfile-kw.gperf"
      {"n_sep_by_space",         tok_n_sep_by_space,         0},
      {""}, {""},
#line 100 "locfile-kw.gperf"
      {"int_n_cs_precedes",      tok_int_n_cs_precedes,      0},
      {""}, {""}, {""},
#line 26 "locfile-kw.gperf"
      {"escape_char",            tok_escape_char,            0},
      {""},
#line 28 "locfile-kw.gperf"
      {"repertoiremap",          tok_repertoiremap,          0},
#line 46 "locfile-kw.gperf"
      {"charclass",              tok_charclass,              0},
#line 43 "locfile-kw.gperf"
      {"print",                  tok_print,                  0},
#line 44 "locfile-kw.gperf"
      {"xdigit",                 tok_xdigit,                 0},
#line 110 "locfile-kw.gperf"
      {"duo_n_cs_precedes",      tok_duo_n_cs_precedes,      0},
#line 127 "locfile-kw.gperf"
      {"thousands_sep",          tok_thousands_sep,          0},
#line 197 "locfile-kw.gperf"
      {"territory",              tok_territory,              0},
#line 36 "locfile-kw.gperf"
      {"digit",                  tok_digit,                  0},
      {""}, {""},
#line 92 "locfile-kw.gperf"
      {"p_cs_precedes",          tok_p_cs_precedes,          0},
      {""}, {""},
#line 62 "locfile-kw.gperf"
      {"script",                 tok_script,                 0},
#line 29 "locfile-kw.gperf"
      {"include",                tok_include,                0},
      {""},
#line 78 "locfile-kw.gperf"
      {"else",                   tok_else,                   0},
#line 184 "locfile-kw.gperf"
      {"int_select",             tok_int_select,             0},
      {""}, {""}, {""},
#line 132 "locfile-kw.gperf"
      {"week",                   tok_week,                   0},
#line 33 "locfile-kw.gperf"
      {"upper",                  tok_upper,                  0},
      {""}, {""},
#line 194 "locfile-kw.gperf"
      {"tel",                    tok_tel,                    0},
#line 93 "locfile-kw.gperf"
      {"p_sep_by_space",         tok_p_sep_by_space,         0},
#line 160 "locfile-kw.gperf"
      {"width",                  tok_width,                  0},
      {""},
#line 98 "locfile-kw.gperf"
      {"int_p_cs_precedes",      tok_int_p_cs_precedes,      0},
      {""}, {""},
#line 41 "locfile-kw.gperf"
      {"punct",                  tok_punct,                  0},
      {""}, {""},
#line 101 "locfile-kw.gperf"
      {"int_n_sep_by_space",     tok_int_n_sep_by_space,     0},
      {""}, {""}, {""},
#line 108 "locfile-kw.gperf"
      {"duo_p_cs_precedes",      tok_duo_p_cs_precedes,      0},
#line 48 "locfile-kw.gperf"
      {"charconv",               tok_charconv,               0},
      {""},
#line 47 "locfile-kw.gperf"
      {"class",                  tok_class,                  0},
#line 114 "locfile-kw.gperf"
      {"duo_int_n_cs_precedes",  tok_duo_int_n_cs_precedes,  0},
#line 115 "locfile-kw.gperf"
      {"duo_int_n_sep_by_space", tok_duo_int_n_sep_by_space, 0},
#line 111 "locfile-kw.gperf"
      {"duo_n_sep_by_space",     tok_duo_n_sep_by_space,     0},
#line 119 "locfile-kw.gperf"
      {"duo_int_n_sign_posn",    tok_duo_int_n_sign_posn,    0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""},
#line 58 "locfile-kw.gperf"
      {"section-symbol",         tok_section_symbol,         0},
#line 185 "locfile-kw.gperf"
      {"int_prefix",             tok_int_prefix,             0},
      {""}, {""}, {""}, {""},
#line 42 "locfile-kw.gperf"
      {"graph",                  tok_graph,                  0},
      {""}, {""},
#line 99 "locfile-kw.gperf"
      {"int_p_sep_by_space",     tok_int_p_sep_by_space,     0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 112 "locfile-kw.gperf"
      {"duo_int_p_cs_precedes",  tok_duo_int_p_cs_precedes,  0},
#line 113 "locfile-kw.gperf"
      {"duo_int_p_sep_by_space", tok_duo_int_p_sep_by_space, 0},
#line 109 "locfile-kw.gperf"
      {"duo_p_sep_by_space",     tok_duo_p_sep_by_space,     0},
#line 118 "locfile-kw.gperf"
      {"duo_int_p_sign_posn",    tok_duo_int_p_sign_posn,    0},
#line 157 "locfile-kw.gperf"
      {"nostr",                  tok_nostr,                  0},
      {""}, {""},
#line 140 "locfile-kw.gperf"
      {"era",                    tok_era,                    0},
      {""},
#line 84 "locfile-kw.gperf"
      {"currency_symbol",        tok_currency_symbol,        0},
      {""},
#line 167 "locfile-kw.gperf"
      {"name_ms",                tok_name_ms,                0},
#line 165 "locfile-kw.gperf"
      {"name_mrs",               tok_name_mrs,               0},
#line 166 "locfile-kw.gperf"
      {"name_miss",              tok_name_miss,              0},
#line 83 "locfile-kw.gperf"
      {"int_curr_symbol",        tok_int_curr_symbol,        0},
#line 190 "locfile-kw.gperf"
      {"source",                 tok_source,                 0},
#line 164 "locfile-kw.gperf"
      {"name_mr",                tok_name_mr,                0},
#line 163 "locfile-kw.gperf"
      {"name_gen",               tok_name_gen,               0},
#line 202 "locfile-kw.gperf"
      {"date",                   tok_date,                   0},
      {""}, {""},
#line 191 "locfile-kw.gperf"
      {"address",                tok_address,                0},
#line 162 "locfile-kw.gperf"
      {"name_fmt",               tok_name_fmt,               0},
#line 32 "locfile-kw.gperf"
      {"copy",                   tok_copy,                   0},
#line 103 "locfile-kw.gperf"
      {"int_n_sign_posn",        tok_int_n_sign_posn,        0},
      {""}, {""},
#line 131 "locfile-kw.gperf"
      {"day",                    tok_day,                    0},
#line 105 "locfile-kw.gperf"
      {"duo_currency_symbol",    tok_duo_currency_symbol,    0},
      {""}, {""}, {""},
#line 150 "locfile-kw.gperf"
      {"date_fmt",               tok_date_fmt,               0},
#line 64 "locfile-kw.gperf"
      {"order_end",              tok_order_end,              0},
#line 117 "locfile-kw.gperf"
      {"duo_n_sign_posn",        tok_duo_n_sign_posn,        0},
      {""},
#line 170 "locfile-kw.gperf"
      {"country_name",           tok_country_name,           0},
#line 71 "locfile-kw.gperf"
      {"reorder-after",          tok_reorder_after,          0},
      {""}, {""},
#line 155 "locfile-kw.gperf"
      {"noexpr",                 tok_noexpr,                 0},
#line 50 "locfile-kw.gperf"
      {"tolower",                tok_tolower,                0},
#line 198 "locfile-kw.gperf"
      {"audience",               tok_audience,               0},
      {""}, {""}, {""},
#line 49 "locfile-kw.gperf"
      {"toupper",                tok_toupper,                0},
#line 68 "locfile-kw.gperf"
      {"position",               tok_position,               0},
      {""},
#line 40 "locfile-kw.gperf"
      {"cntrl",                  tok_cntrl,                  0},
      {""},
#line 27 "locfile-kw.gperf"
      {"comment_char",           tok_comment_char,           0},
#line 88 "locfile-kw.gperf"
      {"positive_sign",          tok_positive_sign,          0},
      {""}, {""}, {""}, {""},
#line 61 "locfile-kw.gperf"
      {"symbol-equivalence",     tok_symbol_equivalence,     0},
      {""},
#line 102 "locfile-kw.gperf"
      {"int_p_sign_posn",        tok_int_p_sign_posn,        0},
#line 175 "locfile-kw.gperf"
      {"country_car",            tok_country_car,            0},
      {""}, {""},
#line 104 "locfile-kw.gperf"
      {"duo_int_curr_symbol",    tok_duo_int_curr_symbol,    0},
      {""}, {""},
#line 135 "locfile-kw.gperf"
      {"d_t_fmt",                tok_d_t_fmt,                0},
      {""}, {""},
#line 116 "locfile-kw.gperf"
      {"duo_p_sign_posn",        tok_duo_p_sign_posn,        0},
#line 187 "locfile-kw.gperf"
      {"measurement",            tok_measurement,            0},
#line 176 "locfile-kw.gperf"
      {"country_isbn",           tok_country_isbn,           0},
#line 37 "locfile-kw.gperf"
      {"outdigit",               tok_outdigit,               0},
      {""}, {""},
#line 143 "locfile-kw.gperf"
      {"era_d_t_fmt",            tok_era_d_t_fmt,            0},
      {""}, {""}, {""},
#line 34 "locfile-kw.gperf"
      {"lower",                  tok_lower,                  0},
#line 183 "locfile-kw.gperf"
      {"tel_dom_fmt",            tok_tel_dom_fmt,            0},
#line 171 "locfile-kw.gperf"
      {"country_post",           tok_country_post,           0},
#line 148 "locfile-kw.gperf"
      {"cal_direction",          tok_cal_direction,          0},
      {""},
#line 139 "locfile-kw.gperf"
      {"t_fmt_ampm",             tok_t_fmt_ampm,             0},
#line 91 "locfile-kw.gperf"
      {"frac_digits",            tok_frac_digits,            0},
      {""}, {""},
#line 177 "locfile-kw.gperf"
      {"lang_name",              tok_lang_name,              0},
#line 90 "locfile-kw.gperf"
      {"int_frac_digits",        tok_int_frac_digits,        0},
      {""},
#line 121 "locfile-kw.gperf"
      {"uno_valid_to",           tok_uno_valid_to,           0},
#line 126 "locfile-kw.gperf"
      {"decimal_point",          tok_decimal_point,          0},
      {""},
#line 133 "locfile-kw.gperf"
      {"abmon",                  tok_abmon,                  0},
      {""}, {""}, {""}, {""},
#line 107 "locfile-kw.gperf"
      {"duo_frac_digits",        tok_duo_frac_digits,        0},
#line 182 "locfile-kw.gperf"
      {"tel_int_fmt",            tok_tel_int_fmt,            0},
#line 123 "locfile-kw.gperf"
      {"duo_valid_to",           tok_duo_valid_to,           0},
#line 146 "locfile-kw.gperf"
      {"first_weekday",          tok_first_weekday,          0},
      {""},
#line 130 "locfile-kw.gperf"
      {"abday",                  tok_abday,                  0},
      {""},
#line 200 "locfile-kw.gperf"
      {"abbreviation",           tok_abbreviation,           0},
#line 147 "locfile-kw.gperf"
      {"first_workday",          tok_first_workday,          0},
      {""}, {""},
#line 97 "locfile-kw.gperf"
      {"n_sign_posn",            tok_n_sign_posn,            0},
      {""}, {""}, {""},
#line 145 "locfile-kw.gperf"
      {"alt_digits",             tok_alt_digits,             0},
      {""}, {""},
#line 128 "locfile-kw.gperf"
      {"grouping",               tok_grouping,               0},
      {""},
#line 45 "locfile-kw.gperf"
      {"blank",                  tok_blank,                  0},
      {""}, {""},
#line 196 "locfile-kw.gperf"
      {"language",               tok_language,               0},
#line 120 "locfile-kw.gperf"
      {"uno_valid_from",         tok_uno_valid_from,         0},
      {""},
#line 199 "locfile-kw.gperf"
      {"application",            tok_application,            0},
      {""},
#line 80 "locfile-kw.gperf"
      {"elifndef",               tok_elifndef,               0},
      {""}, {""}, {""}, {""}, {""},
#line 122 "locfile-kw.gperf"
      {"duo_valid_from",         tok_duo_valid_from,         0},
#line 57 "locfile-kw.gperf"
      {"coll_weight_max",        tok_coll_weight_max,        0},
      {""},
#line 79 "locfile-kw.gperf"
      {"elifdef",                tok_elifdef,                0},
#line 67 "locfile-kw.gperf"
      {"backward",               tok_backward,               0},
#line 106 "locfile-kw.gperf"
      {"duo_int_frac_digits",    tok_duo_int_frac_digits,    0},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 96 "locfile-kw.gperf"
      {"p_sign_posn",            tok_p_sign_posn,            0},
      {""},
#line 203 "locfile-kw.gperf"
      {"category",               tok_category,               0},
      {""}, {""}, {""}, {""},
#line 134 "locfile-kw.gperf"
      {"mon",                    tok_mon,                    0},
      {""},
#line 124 "locfile-kw.gperf"
      {"conversion_rate",        tok_conversion_rate,        0},
      {""}, {""}, {""}, {""}, {""},
#line 63 "locfile-kw.gperf"
      {"order_start",            tok_order_start,            0},
      {""}, {""}, {""}, {""}, {""},
#line 178 "locfile-kw.gperf"
      {"lang_ab",                tok_lang_ab,                0},
#line 180 "locfile-kw.gperf"
      {"lang_lib",               tok_lang_lib,               0},
      {""}, {""}, {""},
#line 192 "locfile-kw.gperf"
      {"contact",                tok_contact,                0},
      {""}, {""}, {""},
#line 173 "locfile-kw.gperf"
      {"country_ab3",            tok_country_ab3,            0},
      {""}, {""}, {""},
#line 193 "locfile-kw.gperf"
      {"email",                  tok_email,                  0},
#line 172 "locfile-kw.gperf"
      {"country_ab2",            tok_country_ab2,            0},
      {""}, {""}, {""},
#line 55 "locfile-kw.gperf"
      {"default_missing",        tok_default_missing,        0},
      {""}, {""},
#line 195 "locfile-kw.gperf"
      {"fax",                    tok_fax,                    0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 174 "locfile-kw.gperf"
      {"country_num",            tok_country_num,            0},
      {""}, {""}, {""}, {""}, {""}, {""},
#line 51 "locfile-kw.gperf"
      {"map",                    tok_map,                    0},
#line 65 "locfile-kw.gperf"
      {"from",                   tok_from,                   0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 86 "locfile-kw.gperf"
      {"mon_thousands_sep",      tok_mon_thousands_sep,      0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""},
#line 81 "locfile-kw.gperf"
      {"endif",                  tok_endif,                  0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 151 "locfile-kw.gperf"
      {"alt_mon",                tok_alt_mon,                0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 76 "locfile-kw.gperf"
      {"undef",                  tok_undef,                  0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 59 "locfile-kw.gperf"
      {"collating-element",      tok_collating_element,      0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 152 "locfile-kw.gperf"
      {"ab_alt_mon",             tok_ab_alt_mon,             0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 66 "locfile-kw.gperf"
      {"forward",                tok_forward,                0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""},
#line 85 "locfile-kw.gperf"
      {"mon_decimal_point",      tok_mon_decimal_point,      0},
      {""}, {""},
#line 169 "locfile-kw.gperf"
      {"postal_fmt",             tok_postal_fmt,             0},
      {""}, {""}, {""}, {""}, {""},
#line 60 "locfile-kw.gperf"
      {"collating-symbol",       tok_collating_symbol,       0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 35 "locfile-kw.gperf"
      {"alpha",                  tok_alpha,                  0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""},
#line 38 "locfile-kw.gperf"
      {"alnum",                  tok_alnum,                  0},
      {""},
#line 87 "locfile-kw.gperf"
      {"mon_grouping",           tok_mon_grouping,           0},
      {""},
#line 179 "locfile-kw.gperf"
      {"lang_term",              tok_lang_term,              0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""},
#line 77 "locfile-kw.gperf"
      {"ifdef",                  tok_ifdef,                  0},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""}, {""},
      {""}, {""}, {""}, {""},
#line 138 "locfile-kw.gperf"
      {"am_pm",                  tok_am_pm,                  0}
    };

  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      register unsigned int key = hash (str, len);

      if (key <= MAX_HASH_VALUE)
        {
          register const char *s = wordlist[key].name;

          if (*str == *s && !strncmp (str + 1, s + 1, len - 1) && s[len] == '\0')
            return &wordlist[key];
        }
    }
  return 0;
}
