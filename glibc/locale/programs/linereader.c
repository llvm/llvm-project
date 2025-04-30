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
   along with this program; if not, see <https://www.gnu.org/licenses/>.  */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <libintl.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "localedef.h"
#include "charmap.h"
#include "error.h"
#include "linereader.h"
#include "locfile.h"

/* Prototypes for local functions.  */
static struct token *get_toplvl_escape (struct linereader *lr);
static struct token *get_symname (struct linereader *lr);
static struct token *get_ident (struct linereader *lr);
static struct token *get_string (struct linereader *lr,
				 const struct charmap_t *charmap,
				 struct localedef_t *locale,
				 const struct repertoire_t *repertoire,
				 int verbose);


struct linereader *
lr_open (const char *fname, kw_hash_fct_t hf)
{
  FILE *fp;

  if (fname == NULL || strcmp (fname, "-") == 0
      || strcmp (fname, "/dev/stdin") == 0)
    return lr_create (stdin, "<stdin>", hf);
  else
    {
      fp = fopen (fname, "rm");
      if (fp == NULL)
	return NULL;
      return lr_create (fp, fname, hf);
    }
}

struct linereader *
lr_create (FILE *fp, const char *fname, kw_hash_fct_t hf)
{
  struct linereader *result;
  int n;

  result = (struct linereader *) xmalloc (sizeof (*result));

  result->fp = fp;
  result->fname = xstrdup (fname);
  result->buf = NULL;
  result->bufsize = 0;
  result->lineno = 1;
  result->idx = 0;
  result->comment_char = '#';
  result->escape_char = '\\';
  result->translate_strings = 1;
  result->return_widestr = 0;

  n = getdelim (&result->buf, &result->bufsize, '\n', result->fp);
  if (n < 0)
    {
      int save = errno;
      fclose (result->fp);
      free ((char *) result->fname);
      free (result);
      errno = save;
      return NULL;
    }

  if (n > 1 && result->buf[n - 2] == '\\' && result->buf[n - 1] == '\n')
    n -= 2;

  result->buf[n] = '\0';
  result->bufact = n;
  result->hash_fct = hf;

  return result;
}


int
lr_eof (struct linereader *lr)
{
  return lr->bufact = 0;
}


void
lr_ignore_rest (struct linereader *lr, int verbose)
{
  if (verbose)
    {
      while (isspace (lr->buf[lr->idx]) && lr->buf[lr->idx] != '\n'
	     && lr->buf[lr->idx] != lr->comment_char)
	if (lr->buf[lr->idx] == '\0')
	  {
	    if (lr_next (lr) < 0)
	      return;
	  }
	else
	  ++lr->idx;

      if (lr->buf[lr->idx] != '\n' && ! feof (lr->fp)
	  && lr->buf[lr->idx] != lr->comment_char)
	lr_error (lr, _("trailing garbage at end of line"));
    }

  /* Ignore continued line.  */
  while (lr->bufact > 0 && lr->buf[lr->bufact - 1] != '\n')
    if (lr_next (lr) < 0)
      break;

  lr->idx = lr->bufact;
}


void
lr_close (struct linereader *lr)
{
  fclose (lr->fp);
  free (lr->buf);
  free (lr);
}


int
lr_next (struct linereader *lr)
{
  int n;

  n = getdelim (&lr->buf, &lr->bufsize, '\n', lr->fp);
  if (n < 0)
    return -1;

  ++lr->lineno;

  if (n > 1 && lr->buf[n - 2] == lr->escape_char && lr->buf[n - 1] == '\n')
    {
#if 0
      /* XXX Is this correct?  */
      /* An escaped newline character is substituted with a single <SP>.  */
      --n;
      lr->buf[n - 1] = ' ';
#else
      n -= 2;
#endif
    }

  lr->buf[n] = '\0';
  lr->bufact = n;
  lr->idx = 0;

  return 0;
}


/* Defined in error.c.  */
/* This variable is incremented each time `error' is called.  */
extern unsigned int error_message_count;

/* The calling program should define program_name and set it to the
   name of the executing program.  */
extern char *program_name;


struct token *
lr_token (struct linereader *lr, const struct charmap_t *charmap,
	  struct localedef_t *locale, const struct repertoire_t *repertoire,
	  int verbose)
{
  int ch;

  while (1)
    {
      do
	{
	  ch = lr_getc (lr);

	  if (ch == EOF)
	    {
	      lr->token.tok = tok_eof;
	      return &lr->token;
	    };

	  if (ch == '\n')
	    {
	      lr->token.tok = tok_eol;
	      return &lr->token;
	    }
	}
      while (isspace (ch));

      if (ch != lr->comment_char)
	break;

      /* Is there an newline at the end of the buffer?  */
      if (lr->buf[lr->bufact - 1] != '\n')
	{
	  /* No.  Some people want this to mean that only the line in
	     the file not the logical, concatenated line is ignored.
	     Let's try this.  */
	  lr->idx = lr->bufact;
	  continue;
	}

      /* Ignore rest of line.  */
      lr_ignore_rest (lr, 0);
      lr->token.tok = tok_eol;
      return &lr->token;
    }

  /* Match escape sequences.  */
  if (ch == lr->escape_char)
    return get_toplvl_escape (lr);

  /* Match ellipsis.  */
  if (ch == '.')
    {
      if (strncmp (&lr->buf[lr->idx], "...(2)....", 10) == 0)
	{
	  int cnt;
	  for (cnt = 0; cnt < 10; ++cnt)
	    lr_getc (lr);
	  lr->token.tok = tok_ellipsis4_2;
	  return &lr->token;
	}
      if (strncmp (&lr->buf[lr->idx], "...", 3) == 0)
	{
	  lr_getc (lr);
	  lr_getc (lr);
	  lr_getc (lr);
	  lr->token.tok = tok_ellipsis4;
	  return &lr->token;
	}
      if (strncmp (&lr->buf[lr->idx], "..", 2) == 0)
	{
	  lr_getc (lr);
	  lr_getc (lr);
	  lr->token.tok = tok_ellipsis3;
	  return &lr->token;
	}
      if (strncmp (&lr->buf[lr->idx], ".(2)..", 6) == 0)
	{
	  int cnt;
	  for (cnt = 0; cnt < 6; ++cnt)
	    lr_getc (lr);
	  lr->token.tok = tok_ellipsis2_2;
	  return &lr->token;
	}
      if (lr->buf[lr->idx] == '.')
	{
	  lr_getc (lr);
	  lr->token.tok = tok_ellipsis2;
	  return &lr->token;
	}
    }

  switch (ch)
    {
    case '<':
      return get_symname (lr);

    case '0' ... '9':
      lr->token.tok = tok_number;
      lr->token.val.num = ch - '0';

      while (isdigit (ch = lr_getc (lr)))
	{
	  lr->token.val.num *= 10;
	  lr->token.val.num += ch - '0';
	}
      if (isalpha (ch))
	lr_error (lr, _("garbage at end of number"));
      lr_ungetn (lr, 1);

      return &lr->token;

    case ';':
      lr->token.tok = tok_semicolon;
      return &lr->token;

    case ',':
      lr->token.tok = tok_comma;
      return &lr->token;

    case '(':
      lr->token.tok = tok_open_brace;
      return &lr->token;

    case ')':
      lr->token.tok = tok_close_brace;
      return &lr->token;

    case '"':
      return get_string (lr, charmap, locale, repertoire, verbose);

    case '-':
      ch = lr_getc (lr);
      if (ch == '1')
	{
	  lr->token.tok = tok_minus1;
	  return &lr->token;
	}
      lr_ungetn (lr, 2);
      break;
    }

  return get_ident (lr);
}


static struct token *
get_toplvl_escape (struct linereader *lr)
{
  /* This is supposed to be a numeric value.  We return the
     numerical value and the number of bytes.  */
  size_t start_idx = lr->idx - 1;
  unsigned char *bytes = lr->token.val.charcode.bytes;
  size_t nbytes = 0;
  int ch;

  do
    {
      unsigned int byte = 0;
      unsigned int base = 8;

      ch = lr_getc (lr);

      if (ch == 'd')
	{
	  base = 10;
	  ch = lr_getc (lr);
	}
      else if (ch == 'x')
	{
	  base = 16;
	  ch = lr_getc (lr);
	}

      if ((base == 16 && !isxdigit (ch))
	  || (base != 16 && (ch < '0' || ch >= (int) ('0' + base))))
	{
	esc_error:
	  lr->token.val.str.startmb = &lr->buf[start_idx];

	  while (ch != EOF && !isspace (ch))
	    ch = lr_getc (lr);
	  lr->token.val.str.lenmb = lr->idx - start_idx;

	  lr->token.tok = tok_error;
	  return &lr->token;
	}

      if (isdigit (ch))
	byte = ch - '0';
      else
	byte = tolower (ch) - 'a' + 10;

      ch = lr_getc (lr);
      if ((base == 16 && !isxdigit (ch))
	  || (base != 16 && (ch < '0' || ch >= (int) ('0' + base))))
	goto esc_error;

      byte *= base;
      if (isdigit (ch))
	byte += ch - '0';
      else
	byte += tolower (ch) - 'a' + 10;

      ch = lr_getc (lr);
      if (base != 16 && isdigit (ch))
	{
	  byte *= base;
	  byte += ch - '0';

	  ch = lr_getc (lr);
	}

      bytes[nbytes++] = byte;
    }
  while (ch == lr->escape_char
	 && nbytes < (int) sizeof (lr->token.val.charcode.bytes));

  if (!isspace (ch))
    lr_error (lr, _("garbage at end of character code specification"));

  lr_ungetn (lr, 1);

  lr->token.tok = tok_charcode;
  lr->token.val.charcode.nbytes = nbytes;

  return &lr->token;
}


#define ADDC(ch) \
  do									      \
    {									      \
      if (bufact == bufmax)						      \
	{								      \
	  bufmax *= 2;							      \
	  buf = xrealloc (buf, bufmax);					      \
	}								      \
      buf[bufact++] = (ch);						      \
    }									      \
  while (0)


#define ADDS(s, l) \
  do									      \
    {									      \
      size_t _l = (l);							      \
      if (bufact + _l > bufmax)						      \
	{								      \
	  if (bufact < _l)						      \
	    bufact = _l;						      \
	  bufmax *= 2;							      \
	  buf = xrealloc (buf, bufmax);					      \
	}								      \
      memcpy (&buf[bufact], s, _l);					      \
      bufact += _l;							      \
    }									      \
  while (0)


#define ADDWC(ch) \
  do									      \
    {									      \
      if (buf2act == buf2max)						      \
	{								      \
	  buf2max *= 2;							      \
	  buf2 = xrealloc (buf2, buf2max * 4);				      \
	}								      \
      buf2[buf2act++] = (ch);						      \
    }									      \
  while (0)


static struct token *
get_symname (struct linereader *lr)
{
  /* Symbol in brackets.  We must distinguish three kinds:
     1. reserved words
     2. ISO 10646 position values
     3. all other.  */
  char *buf;
  size_t bufact = 0;
  size_t bufmax = 56;
  const struct keyword_t *kw;
  int ch;

  buf = (char *) xmalloc (bufmax);

  do
    {
      ch = lr_getc (lr);
      if (ch == lr->escape_char)
	{
	  int c2 = lr_getc (lr);
	  ADDC (c2);

	  if (c2 == '\n')
	    ch = '\n';
	}
      else
	ADDC (ch);
    }
  while (ch != '>' && ch != '\n');

  if (ch == '\n')
    lr_error (lr, _("unterminated symbolic name"));

  /* Test for ISO 10646 position value.  */
  if (buf[0] == 'U' && (bufact == 6 || bufact == 10))
    {
      char *cp = buf + 1;
      while (cp < &buf[bufact - 1] && isxdigit (*cp))
	++cp;

      if (cp == &buf[bufact - 1])
	{
	  /* Yes, it is.  */
	  lr->token.tok = tok_ucs4;
	  lr->token.val.ucs4 = strtoul (buf + 1, NULL, 16);

	  return &lr->token;
	}
    }

  /* It is a symbolic name.  Test for reserved words.  */
  kw = lr->hash_fct (buf, bufact - 1);

  if (kw != NULL && kw->symname_or_ident == 1)
    {
      lr->token.tok = kw->token;
      free (buf);
    }
  else
    {
      lr->token.tok = tok_bsymbol;

      buf = xrealloc (buf, bufact + 1);
      buf[bufact] = '\0';

      lr->token.val.str.startmb = buf;
      lr->token.val.str.lenmb = bufact - 1;
    }

  return &lr->token;
}


static struct token *
get_ident (struct linereader *lr)
{
  char *buf;
  size_t bufact;
  size_t bufmax = 56;
  const struct keyword_t *kw;
  int ch;

  buf = xmalloc (bufmax);
  bufact = 0;

  ADDC (lr->buf[lr->idx - 1]);

  while (!isspace ((ch = lr_getc (lr))) && ch != '"' && ch != ';'
	 && ch != '<' && ch != ',' && ch != EOF)
    {
      if (ch == lr->escape_char)
	{
	  ch = lr_getc (lr);
	  if (ch == '\n' || ch == EOF)
	    {
	      lr_error (lr, _("invalid escape sequence"));
	      break;
	    }
	}
      ADDC (ch);
    }

  lr_ungetc (lr, ch);

  kw = lr->hash_fct (buf, bufact);

  if (kw != NULL && kw->symname_or_ident == 0)
    {
      lr->token.tok = kw->token;
      free (buf);
    }
  else
    {
      lr->token.tok = tok_ident;

      buf = xrealloc (buf, bufact + 1);
      buf[bufact] = '\0';

      lr->token.val.str.startmb = buf;
      lr->token.val.str.lenmb = bufact;
    }

  return &lr->token;
}


static struct token *
get_string (struct linereader *lr, const struct charmap_t *charmap,
	    struct localedef_t *locale, const struct repertoire_t *repertoire,
	    int verbose)
{
  int return_widestr = lr->return_widestr;
  char *buf;
  wchar_t *buf2 = NULL;
  size_t bufact;
  size_t bufmax = 56;

  /* We must return two different strings.  */
  buf = xmalloc (bufmax);
  bufact = 0;

  /* We know it'll be a string.  */
  lr->token.tok = tok_string;

  /* If we need not translate the strings (i.e., expand <...> parts)
     we can run a simple loop.  */
  if (!lr->translate_strings)
    {
      int ch;

      buf2 = NULL;
      while ((ch = lr_getc (lr)) != '"' && ch != '\n' && ch != EOF)
	ADDC (ch);

      /* Catch errors with trailing escape character.  */
      if (bufact > 0 && buf[bufact - 1] == lr->escape_char
	  && (bufact == 1 || buf[bufact - 2] != lr->escape_char))
	{
	  lr_error (lr, _("illegal escape sequence at end of string"));
	  --bufact;
	}
      else if (ch == '\n' || ch == EOF)
	lr_error (lr, _("unterminated string"));

      ADDC ('\0');
    }
  else
    {
      int illegal_string = 0;
      size_t buf2act = 0;
      size_t buf2max = 56 * sizeof (uint32_t);
      int ch;

      /* We have to provide the wide character result as well.  */
      if (return_widestr)
	buf2 = xmalloc (buf2max);

      /* Read until the end of the string (or end of the line or file).  */
      while ((ch = lr_getc (lr)) != '"' && ch != '\n' && ch != EOF)
	{
	  size_t startidx;
	  uint32_t wch;
	  struct charseq *seq;

	  if (ch != '<')
	    {
	      /* The standards leave it up to the implementation to decide
		 what to do with character which stand for themself.  We
		 could jump through hoops to find out the value relative to
		 the charmap and the repertoire map, but instead we leave
		 it up to the locale definition author to write a better
		 definition.  We assume here that every character which
		 stands for itself is encoded using ISO 8859-1.  Using the
		 escape character is allowed.  */
	      if (ch == lr->escape_char)
		{
		  ch = lr_getc (lr);
		  if (ch == '\n' || ch == EOF)
		    break;
		}

	      ADDC (ch);
	      if (return_widestr)
		ADDWC ((uint32_t) ch);

	      continue;
	    }

	  /* Now we have to search for the end of the symbolic name, i.e.,
	     the closing '>'.  */
	  startidx = bufact;
	  while ((ch = lr_getc (lr)) != '>' && ch != '\n' && ch != EOF)
	    {
	      if (ch == lr->escape_char)
		{
		  ch = lr_getc (lr);
		  if (ch == '\n' || ch == EOF)
		    break;
		}
	      ADDC (ch);
	    }
	  if (ch == '\n' || ch == EOF)
	    /* Not a correct string.  */
	    break;
	  if (bufact == startidx)
	    {
	      /* <> is no correct name.  Ignore it and also signal an
		 error.  */
	      illegal_string = 1;
	      continue;
	    }

	  /* It might be a Uxxxx symbol.  */
	  if (buf[startidx] == 'U'
	      && (bufact - startidx == 5 || bufact - startidx == 9))
	    {
	      char *cp = buf + startidx + 1;
	      while (cp < &buf[bufact] && isxdigit (*cp))
		++cp;

	      if (cp == &buf[bufact])
		{
		  char utmp[10];

		  /* Yes, it is.  */
		  ADDC ('\0');
		  wch = strtoul (buf + startidx + 1, NULL, 16);

		  /* Now forget about the name we just added.  */
		  bufact = startidx;

		  if (return_widestr)
		    ADDWC (wch);

		  /* See whether the charmap contains the Uxxxxxxxx names.  */
		  snprintf (utmp, sizeof (utmp), "U%08X", wch);
		  seq = charmap_find_value (charmap, utmp, 9);

		  if (seq == NULL)
		    {
		     /* No, this isn't the case.  Now determine from
			the repertoire the name of the character and
			find it in the charmap.  */
		      if (repertoire != NULL)
			{
			  const char *symbol;

			  symbol = repertoire_find_symbol (repertoire, wch);

			  if (symbol != NULL)
			    seq = charmap_find_value (charmap, symbol,
						      strlen (symbol));
			}

		      if (seq == NULL)
			{
#ifndef NO_TRANSLITERATION
			  /* Transliterate if possible.  */
			  if (locale != NULL)
			    {
			      uint32_t *translit;

			      if ((locale->avail & CTYPE_LOCALE) == 0)
				{
				  /* Load the CTYPE data now.  */
				  int old_needed = locale->needed;

				  locale->needed = 0;
				  locale = load_locale (LC_CTYPE,
							locale->name,
							locale->repertoire_name,
							charmap, locale);
				  locale->needed = old_needed;
				}

			      if ((locale->avail & CTYPE_LOCALE) != 0
				  && ((translit = find_translit (locale,
								 charmap, wch))
				      != NULL))
				/* The CTYPE data contains a matching
				   transliteration.  */
				{
				  int i;

				  for (i = 0; translit[i] != 0; ++i)
				    {
				      char utmp[10];

				      snprintf (utmp, sizeof (utmp), "U%08X",
						translit[i]);
				      seq = charmap_find_value (charmap, utmp,
								9);
				      assert (seq != NULL);
				      ADDS (seq->bytes, seq->nbytes);
				    }

				  continue;
				}
			    }
#endif	/* NO_TRANSLITERATION */

			  /* Not a known name.  */
			  illegal_string = 1;
			}
		    }

		  if (seq != NULL)
		    ADDS (seq->bytes, seq->nbytes);

		  continue;
		}
	    }

	  /* We now have the symbolic name in buf[startidx] to
	     buf[bufact-1].  Now find out the value for this character
	     in the charmap as well as in the repertoire map (in this
	     order).  */
	  seq = charmap_find_value (charmap, &buf[startidx],
				    bufact - startidx);

	  if (seq == NULL)
	    {
	      /* This name is not in the charmap.  */
	      lr_error (lr, _("symbol `%.*s' not in charmap"),
			(int) (bufact - startidx), &buf[startidx]);
	      illegal_string = 1;
	    }

	  if (return_widestr)
	    {
	      /* Now the same for the multibyte representation.  */
	      if (seq != NULL && seq->ucs4 != UNINITIALIZED_CHAR_VALUE)
		wch = seq->ucs4;
	      else
		{
		  wch = repertoire_find_value (repertoire, &buf[startidx],
					       bufact - startidx);
		  if (seq != NULL)
		    seq->ucs4 = wch;
		}

	      if (wch == ILLEGAL_CHAR_VALUE)
		{
		  /* This name is not in the repertoire map.  */
		  lr_error (lr, _("symbol `%.*s' not in repertoire map"),
			    (int) (bufact - startidx), &buf[startidx]);
		  illegal_string = 1;
		}
	      else
		ADDWC (wch);
	    }

	  /* Now forget about the name we just added.  */
	  bufact = startidx;

	  /* And copy the bytes.  */
	  if (seq != NULL)
	    ADDS (seq->bytes, seq->nbytes);
	}

      if (ch == '\n' || ch == EOF)
	{
	  lr_error (lr, _("unterminated string"));
	  illegal_string = 1;
	}

      if (illegal_string)
	{
	  free (buf);
	  free (buf2);
	  lr->token.val.str.startmb = NULL;
	  lr->token.val.str.lenmb = 0;
	  lr->token.val.str.startwc = NULL;
	  lr->token.val.str.lenwc = 0;

	  return &lr->token;
	}

      ADDC ('\0');

      if (return_widestr)
	{
	  ADDWC (0);
	  lr->token.val.str.startwc = xrealloc (buf2,
						buf2act * sizeof (uint32_t));
	  lr->token.val.str.lenwc = buf2act;
	}
    }

  lr->token.val.str.startmb = xrealloc (buf, bufact);
  lr->token.val.str.lenmb = bufact;

  return &lr->token;
}
