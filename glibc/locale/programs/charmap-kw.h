/* ANSI-C code produced by gperf version 3.1 */
/* Command-line: gperf -acCgopt -k'1,2,5,9,$' -L ANSI-C -N charmap_hash charmap-kw.gperf  */

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

#line 1 "charmap-kw.gperf"

/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper, <drepper@gnu.org>.

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
#line 24 "charmap-kw.gperf"
struct keyword_t ;

#define TOTAL_KEYWORDS 17
#define MIN_WORD_LENGTH 3
#define MAX_WORD_LENGTH 14
#define MIN_HASH_VALUE 3
#define MAX_HASH_VALUE 35
/* maximum key range = 33, duplicates = 0 */

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
  static const unsigned char asso_values[] =
    {
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 25, 20,
      15, 10, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36,  5,  0,  0,
       5, 36,  0,  0, 36, 36, 36,  5,  0, 36,
       0, 36,  0, 36,  0, 36, 36,  0, 36, 36,
      36, 36, 36, 36, 36,  0, 36,  5,  0,  0,
       5,  0, 36,  5,  0,  0, 36, 36, 36,  0,
       0,  0,  0,  0,  0,  0,  0,  0, 36, 36,
       0, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36, 36, 36, 36, 36,
      36, 36, 36, 36, 36, 36
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
charmap_hash (register const char *str, register size_t len)
{
  static const struct keyword_t wordlist[] =
    {
      {""}, {""}, {""},
#line 39 "charmap-kw.gperf"
      {"END",             tok_end,             0},
      {""},
#line 40 "charmap-kw.gperf"
      {"WIDTH",           tok_width,           0},
#line 35 "charmap-kw.gperf"
      {"escseq",          tok_escseq,          1},
#line 37 "charmap-kw.gperf"
      {"include",         tok_include,         1},
      {""}, {""},
#line 28 "charmap-kw.gperf"
      {"mb_cur_min",      tok_mb_cur_min,      1},
#line 29 "charmap-kw.gperf"
      {"escape_char",     tok_escape_char,     1},
#line 30 "charmap-kw.gperf"
      {"comment_char",    tok_comment_char,    1},
#line 26 "charmap-kw.gperf"
      {"code_set_name",   tok_code_set_name,   1},
#line 41 "charmap-kw.gperf"
      {"WIDTH_VARIABLE",  tok_width_variable,  0},
#line 27 "charmap-kw.gperf"
      {"mb_cur_max",      tok_mb_cur_max,      1},
#line 36 "charmap-kw.gperf"
      {"addset",          tok_addset,          1},
#line 38 "charmap-kw.gperf"
      {"CHARMAP",         tok_charmap,         0},
#line 42 "charmap-kw.gperf"
      {"WIDTH_DEFAULT",   tok_width_default,   0},
      {""},
#line 34 "charmap-kw.gperf"
      {"g3esc",           tok_g3esc,           1},
      {""}, {""}, {""}, {""},
#line 33 "charmap-kw.gperf"
      {"g2esc",           tok_g2esc,           1},
      {""}, {""}, {""}, {""},
#line 32 "charmap-kw.gperf"
      {"g1esc",           tok_g1esc,           1},
      {""}, {""}, {""}, {""},
#line 31 "charmap-kw.gperf"
      {"g0esc",           tok_g0esc,           1}
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
