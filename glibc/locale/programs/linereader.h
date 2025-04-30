/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifndef _LINEREADER_H
#define _LINEREADER_H 1

#include <ctype.h>
#include <libintl.h>
#include <stdint.h>
#include <stdio.h>

#include "charmap.h"
#include "error.h"
#include "locfile-token.h"
#include "repertoire.h"
#include "record-status.h"

typedef const struct keyword_t *(*kw_hash_fct_t) (const char *, size_t);
struct charset_t;
struct localedef_t;

struct token
{
  enum token_t tok;
  union
  {
    struct
    {
      char *startmb;
      size_t lenmb;
      uint32_t *startwc;
      size_t lenwc;
    } str;
    unsigned long int num;
    struct
    {
      /* This element is sized on the safe expectation that no single
	 character in any character set uses more than 16 bytes.  */
      unsigned char bytes[16];
      int nbytes;
    } charcode;
    uint32_t ucs4;
  } val;
};


struct linereader
{
  FILE *fp;
  const char *fname;
  char *buf;
  size_t bufsize;
  size_t bufact;
  size_t lineno;

  size_t idx;

  char comment_char;
  char escape_char;

  struct token token;

  int translate_strings;
  int return_widestr;

  kw_hash_fct_t hash_fct;
};


/* Functions defined in linereader.c.  */
extern struct linereader *lr_open (const char *fname, kw_hash_fct_t hf);
extern struct linereader *lr_create (FILE *fp, const char *fname,
				     kw_hash_fct_t hf);
extern int lr_eof (struct linereader *lr);
extern void lr_close (struct linereader *lr);
extern int lr_next (struct linereader *lr);
extern struct token *lr_token (struct linereader *lr,
			       const struct charmap_t *charmap,
			       struct localedef_t *locale,
			       const struct repertoire_t *repertoire,
			       int verbose);
extern void lr_ignore_rest (struct linereader *lr, int verbose);


static inline void
__attribute__ ((__format__ (__printf__, 2, 3), nonnull (1, 2)))
lr_error (struct linereader *lr, const char *fmt, ...)
{
  char *str;
  va_list arg;
  struct locale_state ls;
  int ret;

  va_start (arg, fmt);
  ls = push_locale ();

  ret = vasprintf (&str, fmt, arg);
  if (ret == -1)
    abort ();

  pop_locale (ls);
  va_end (arg);

  error_at_line (0, 0, lr->fname, lr->lineno, "%s", str);

  free (str);
}


static inline int
__attribute ((always_inline))
lr_getc (struct linereader *lr)
{
  if (lr->idx == lr->bufact)
    {
      if (lr->bufact != 0)
	if (lr_next (lr) < 0)
	  return EOF;

      if (lr->bufact == 0)
	return EOF;
    }

  return lr->buf[lr->idx] == '\32' ? EOF : lr->buf[lr->idx++];
}


static inline int
__attribute ((always_inline))
lr_ungetc (struct linereader *lr, int ch)
{
  if (lr->idx == 0)
    return -1;

  if (ch != EOF)
    lr->buf[--lr->idx] = ch;
  return 0;
}


static inline int
lr_ungetn (struct linereader *lr, size_t n)
{
  if (lr->idx < n)
    return -1;

  lr->idx -= n;
  return 0;
}


#endif /* linereader.h */
