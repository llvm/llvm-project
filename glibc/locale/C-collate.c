/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1995.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <endian.h>
#include <stdint.h>
#include "localeinfo.h"

static const char collseqmb[] =
{
  '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
  '\x08', '\x09', '\x0a', '\x0b', '\x0c', '\x0d', '\x0e', '\x0f',
  '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
  '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
  '\x20', '\x21', '\x22', '\x23', '\x24', '\x25', '\x26', '\x27',
  '\x28', '\x29', '\x2a', '\x2b', '\x2c', '\x2d', '\x2e', '\x2f',
  '\x30', '\x31', '\x32', '\x33', '\x34', '\x35', '\x36', '\x37',
  '\x38', '\x39', '\x3a', '\x3b', '\x3c', '\x3d', '\x3e', '\x3f',
  '\x40', '\x41', '\x42', '\x43', '\x44', '\x45', '\x46', '\x47',
  '\x48', '\x49', '\x4a', '\x4b', '\x4c', '\x4d', '\x4e', '\x4f',
  '\x50', '\x51', '\x52', '\x53', '\x54', '\x55', '\x56', '\x57',
  '\x58', '\x59', '\x5a', '\x5b', '\x5c', '\x5d', '\x5e', '\x5f',
  '\x60', '\x61', '\x62', '\x63', '\x64', '\x65', '\x66', '\x67',
  '\x68', '\x69', '\x6a', '\x6b', '\x6c', '\x6d', '\x6e', '\x6f',
  '\x70', '\x71', '\x72', '\x73', '\x74', '\x75', '\x76', '\x77',
  '\x78', '\x79', '\x7a', '\x7b', '\x7c', '\x7d', '\x7e', '\x7f',
  '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87',
  '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f',
  '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97',
  '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f',
  '\xa0', '\xa1', '\xa2', '\xa3', '\xa4', '\xa5', '\xa6', '\xa7',
  '\xa8', '\xa9', '\xaa', '\xab', '\xac', '\xad', '\xae', '\xaf',
  '\xb0', '\xb1', '\xb2', '\xb3', '\xb4', '\xb5', '\xb6', '\xb7',
  '\xb8', '\xb9', '\xba', '\xbb', '\xbc', '\xbd', '\xbe', '\xbf',
  '\xc0', '\xc1', '\xc2', '\xc3', '\xc4', '\xc5', '\xc6', '\xc7',
  '\xc8', '\xc9', '\xca', '\xcb', '\xcc', '\xcd', '\xce', '\xcf',
  '\xd0', '\xd1', '\xd2', '\xd3', '\xd4', '\xd5', '\xd6', '\xd7',
  '\xd8', '\xd9', '\xda', '\xdb', '\xdc', '\xdd', '\xde', '\xdf',
  '\xe0', '\xe1', '\xe2', '\xe3', '\xe4', '\xe5', '\xe6', '\xe7',
  '\xe8', '\xe9', '\xea', '\xeb', '\xec', '\xed', '\xee', '\xef',
  '\xf0', '\xf1', '\xf2', '\xf3', '\xf4', '\xf5', '\xf6', '\xf7',
  '\xf8', '\xf9', '\xfa', '\xfb', '\xfc', '\xfd', '\xfe', '\xff'
};

static const uint32_t collseqwc[] =
{
  8, 1, 8, 0x0, 0xff,
  /* 1st-level table */
  6 * sizeof (uint32_t),
  /* 2nd-level table */
  7 * sizeof (uint32_t),
  /* 3rd-level table */
  L'\x00', L'\x01', L'\x02', L'\x03', L'\x04', L'\x05', L'\x06', L'\x07',
  L'\x08', L'\x09', L'\x0a', L'\x0b', L'\x0c', L'\x0d', L'\x0e', L'\x0f',
  L'\x10', L'\x11', L'\x12', L'\x13', L'\x14', L'\x15', L'\x16', L'\x17',
  L'\x18', L'\x19', L'\x1a', L'\x1b', L'\x1c', L'\x1d', L'\x1e', L'\x1f',
  L'\x20', L'\x21', L'\x22', L'\x23', L'\x24', L'\x25', L'\x26', L'\x27',
  L'\x28', L'\x29', L'\x2a', L'\x2b', L'\x2c', L'\x2d', L'\x2e', L'\x2f',
  L'\x30', L'\x31', L'\x32', L'\x33', L'\x34', L'\x35', L'\x36', L'\x37',
  L'\x38', L'\x39', L'\x3a', L'\x3b', L'\x3c', L'\x3d', L'\x3e', L'\x3f',
  L'\x40', L'\x41', L'\x42', L'\x43', L'\x44', L'\x45', L'\x46', L'\x47',
  L'\x48', L'\x49', L'\x4a', L'\x4b', L'\x4c', L'\x4d', L'\x4e', L'\x4f',
  L'\x50', L'\x51', L'\x52', L'\x53', L'\x54', L'\x55', L'\x56', L'\x57',
  L'\x58', L'\x59', L'\x5a', L'\x5b', L'\x5c', L'\x5d', L'\x5e', L'\x5f',
  L'\x60', L'\x61', L'\x62', L'\x63', L'\x64', L'\x65', L'\x66', L'\x67',
  L'\x68', L'\x69', L'\x6a', L'\x6b', L'\x6c', L'\x6d', L'\x6e', L'\x6f',
  L'\x70', L'\x71', L'\x72', L'\x73', L'\x74', L'\x75', L'\x76', L'\x77',
  L'\x78', L'\x79', L'\x7a', L'\x7b', L'\x7c', L'\x7d', L'\x7e', L'\x7f',
  L'\x80', L'\x81', L'\x82', L'\x83', L'\x84', L'\x85', L'\x86', L'\x87',
  L'\x88', L'\x89', L'\x8a', L'\x8b', L'\x8c', L'\x8d', L'\x8e', L'\x8f',
  L'\x90', L'\x91', L'\x92', L'\x93', L'\x94', L'\x95', L'\x96', L'\x97',
  L'\x98', L'\x99', L'\x9a', L'\x9b', L'\x9c', L'\x9d', L'\x9e', L'\x9f',
  L'\xa0', L'\xa1', L'\xa2', L'\xa3', L'\xa4', L'\xa5', L'\xa6', L'\xa7',
  L'\xa8', L'\xa9', L'\xaa', L'\xab', L'\xac', L'\xad', L'\xae', L'\xaf',
  L'\xb0', L'\xb1', L'\xb2', L'\xb3', L'\xb4', L'\xb5', L'\xb6', L'\xb7',
  L'\xb8', L'\xb9', L'\xba', L'\xbb', L'\xbc', L'\xbd', L'\xbe', L'\xbf',
  L'\xc0', L'\xc1', L'\xc2', L'\xc3', L'\xc4', L'\xc5', L'\xc6', L'\xc7',
  L'\xc8', L'\xc9', L'\xca', L'\xcb', L'\xcc', L'\xcd', L'\xce', L'\xcf',
  L'\xd0', L'\xd1', L'\xd2', L'\xd3', L'\xd4', L'\xd5', L'\xd6', L'\xd7',
  L'\xd8', L'\xd9', L'\xda', L'\xdb', L'\xdc', L'\xdd', L'\xde', L'\xdf',
  L'\xe0', L'\xe1', L'\xe2', L'\xe3', L'\xe4', L'\xe5', L'\xe6', L'\xe7',
  L'\xe8', L'\xe9', L'\xea', L'\xeb', L'\xec', L'\xed', L'\xee', L'\xef',
  L'\xf0', L'\xf1', L'\xf2', L'\xf3', L'\xf4', L'\xf5', L'\xf6', L'\xf7',
  L'\xf8', L'\xf9', L'\xfa', L'\xfb', L'\xfc', L'\xfd', L'\xfe', L'\xff'
};

const struct __locale_data _nl_C_LC_COLLATE attribute_hidden =
{
  _nl_C_name,
  NULL, 0, 0,			/* no file mapped */
  { NULL, },			/* no cached data */
  UNDELETABLE,
  0,
  19,
  {
    /* _NL_COLLATE_NRULES */
    { .word = 0 },
    /* _NL_COLLATE_RULESETS */
    { .string = NULL },
    /* _NL_COLLATE_TABLEMB */
    { .string = NULL },
    /* _NL_COLLATE_WEIGHTMB */
    { .string = NULL },
    /* _NL_COLLATE_EXTRAMB */
    { .string = NULL },
    /* _NL_COLLATE_INDIRECTMB */
    { .string = NULL },
    /* _NL_COLLATE_GAP1 */
    { .string = NULL },
    /* _NL_COLLATE_GAP2 */
    { .string = NULL },
    /* _NL_COLLATE_GAP3 */
    { .string = NULL },
    /* _NL_COLLATE_TABLEWC */
    { .string = NULL },
    /* _NL_COLLATE_WEIGHTWC */
    { .string = NULL },
    /* _NL_COLLATE_EXTRAWC */
    { .string = NULL },
    /* _NL_COLLATE_INDIRECTWC */
    { .string = NULL },
    /* _NL_COLLATE_SYMB_HASH_SIZEMB */
    { .string = NULL },
    /* _NL_COLLATE_SYMB_TABLEMB */
    { .string = NULL },
    /* _NL_COLLATE_SYMB_EXTRAMB */
    { .string = NULL },
    /* _NL_COLLATE_COLLSEQMB */
    { .string = collseqmb },
    /* _NL_COLLATE_COLLSEQWC */
    { .string = (const char *) collseqwc },
    /* _NL_COLLATE_CODESET */
    { .string = _nl_C_codeset }
  }
};
