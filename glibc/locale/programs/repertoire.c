/* Copyright (C) 1998-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1998.

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

#include <errno.h>
#include <limits.h>
#include <obstack.h>
#include <search.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>

#include "localedef.h"
#include "linereader.h"
#include "charmap.h"
#include "repertoire.h"
#include "simple-hash.h"


/* Simple keyword hashing for the repertoiremap.  */
static const struct keyword_t *repertoiremap_hash (const char *str,
						   size_t len);
static void repertoire_new_char (struct linereader *lr, hash_table *ht,
				 hash_table *rt, struct obstack *ob,
				 uint32_t value, const char *from,
				 const char *to, int decimal_ellipsis);
static int repertoire_compare (const void *p1, const void *p2);

/* Already known repertoire maps.  */
static void *known;

/* List of repertoire maps which are not available and which have been
   reported to not be.  */
static void *unavailable;


struct repertoire_t *
repertoire_read (const char *filename)
{
  struct linereader *repfile;
  struct repertoire_t *result;
  struct repertoire_t **resultp;
  struct repertoire_t search;
  int state;
  char *from_name = NULL;
  char *to_name = NULL;
  enum token_t ellipsis = tok_none;

  search.name = filename;
  resultp = tfind (&search, &known, &repertoire_compare);
  if (resultp != NULL)
    return *resultp;

  /* Determine path.  */
  repfile = lr_open (filename, repertoiremap_hash);
  if (repfile == NULL)
    {
      if (strchr (filename, '/') == NULL)
	{
	  char *i18npath = getenv ("I18NPATH");
	  if (i18npath != NULL && *i18npath != '\0')
	    {
	      const size_t pathlen = strlen (i18npath);
	      char i18npathbuf[pathlen + 1];
	      char path[strlen (filename) + 1 + pathlen
		        + sizeof ("/repertoiremaps/") - 1];
	      char *next;
	      i18npath = memcpy (i18npathbuf, i18npath, pathlen + 1);

	      while (repfile == NULL
		     && (next = strsep (&i18npath, ":")) != NULL)
		{
		  stpcpy (stpcpy (stpcpy (path, next), "/repertoiremaps/"),
			  filename);

		  repfile = lr_open (path, repertoiremap_hash);

		  if (repfile == NULL)
		    {
		      stpcpy (stpcpy (stpcpy (path, next), "/"), filename);

		      repfile = lr_open (path, repertoiremap_hash);
		    }
		}
	    }

	  if (repfile == NULL)
	    {
	      /* Look in the systems charmap directory.  */
	      char *buf = xmalloc (strlen (filename) + 1
				   + sizeof (REPERTOIREMAP_PATH));

	      stpcpy (stpcpy (stpcpy (buf, REPERTOIREMAP_PATH), "/"),
		      filename);
	      repfile = lr_open (buf, repertoiremap_hash);

	      free (buf);
	    }
	}

      if (repfile == NULL)
	return NULL;
    }

  /* We don't want symbolic names in string to be translated.  */
  repfile->translate_strings = 0;

  /* Allocate room for result.  */
  result = (struct repertoire_t *) xmalloc (sizeof (struct repertoire_t));
  memset (result, '\0', sizeof (struct repertoire_t));

  result->name = xstrdup (filename);

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
  obstack_init (&result->mem_pool);

  if (init_hash (&result->char_table, 256)
      || init_hash (&result->reverse_table, 256)
      || init_hash (&result->seq_table, 256))
    {
      free (result);
      return NULL;
    }

  /* We use a state machine to describe the charmap description file
     format.  */
  state = 1;
  while (1)
    {
      /* What's on?  */
      struct token *now = lr_token (repfile, NULL, NULL, NULL, verbose);
      enum token_t nowtok = now->tok;
      struct token *arg;

      if (nowtok == tok_eof)
	break;

      switch (state)
	{
	case 1:
	  /* We haven't yet read any character definition.  This is where
	     we accept escape_char and comment_char definitions.  */
	  if (nowtok == tok_eol)
	    /* Ignore empty lines.  */
	    continue;

	  if (nowtok == tok_escape_char || nowtok == tok_comment_char)
	    {
	      /* We know that we need an argument.  */
	      arg = lr_token (repfile, NULL, NULL, NULL, verbose);

	      if (arg->tok != tok_ident)
		{
		  lr_error (repfile, _("syntax error in prolog: %s"),
			    _("bad argument"));

		  lr_ignore_rest (repfile, 0);
		  continue;
		}

	      if (arg->val.str.lenmb != 1)
		{
		  lr_error (repfile, _("\
argument to <%s> must be a single character"),
			    nowtok == tok_escape_char ? "escape_char"
						      : "comment_char");

		  lr_ignore_rest (repfile, 0);
		  continue;
		}

	      if (nowtok == tok_escape_char)
		repfile->escape_char = *arg->val.str.startmb;
	      else
		repfile->comment_char = *arg->val.str.startmb;

	      lr_ignore_rest (repfile, 1);
	      continue;
	    }

	  if (nowtok == tok_charids)
	    {
	      lr_ignore_rest (repfile, 1);

	      state = 2;
	      continue;
	    }

	  /* Otherwise we start reading the character definitions.  */
	  state = 2;
	  /* FALLTHROUGH */

	case 2:
	  /* We are now are in the body.  Each line
	     must have the format "%s %s %s\n" or "%s...%s %s %s\n".  */
	  if (nowtok == tok_eol)
	    /* Ignore empty lines.  */
	    continue;

	  if (nowtok == tok_end)
	    {
	      state = 90;
	      continue;
	    }

	  if (nowtok != tok_bsymbol)
	    {
	      lr_error (repfile,
			_("syntax error in repertoire map definition: %s"),
			_("no symbolic name given"));

	      lr_ignore_rest (repfile, 0);
	      continue;
	    }

	  /* If the previous line was not completely correct free the
	     used memory.  */
	  if (from_name != NULL)
	    obstack_free (&result->mem_pool, from_name);

	  from_name = (char *) obstack_copy0 (&result->mem_pool,
					      now->val.str.startmb,
					      now->val.str.lenmb);
	  to_name = NULL;

	  state = 3;
	  continue;

	case 3:
	  /* We have two possibilities: We can see an ellipsis or an
	     encoding value.  */
	  if (nowtok == tok_ellipsis3 || nowtok == tok_ellipsis4
	      || nowtok == tok_ellipsis2)
	    {
	      ellipsis = nowtok;
	      state = 4;
	      continue;
	    }
	  /* FALLTHROUGH */

	case 5:
	  /* We expect a value of the form <Uxxxx> or <Uxxxxxxxx> where
	     the xxx mean a hexadecimal value.  */
	  state = 2;

	  errno = 0;
	  if (nowtok != tok_ucs4)
	    {
	      lr_error (repfile,
			_("syntax error in repertoire map definition: %s"),
			_("no <Uxxxx> or <Uxxxxxxxx> value given"));

	      lr_ignore_rest (repfile, 0);
	      continue;
	    }

	  /* We've found a new valid definition.  */
	  repertoire_new_char (repfile, &result->char_table,
			       &result->reverse_table, &result->mem_pool,
			       now->val.ucs4, from_name, to_name,
			       ellipsis != tok_ellipsis2);

	  /* Ignore the rest of the line.  */
	  lr_ignore_rest (repfile, 0);

	  from_name = NULL;
	  to_name = NULL;

	  continue;

	case 4:
	  if (nowtok != tok_bsymbol)
	    {
	      lr_error (repfile,
			_("syntax error in repertoire map definition: %s"),
			_("no symbolic name given for end of range"));

	      lr_ignore_rest (repfile, 0);
	      state = 2;
	      continue;
	    }

	  /* Copy the to-name in a safe place.  */
	  to_name = (char *) obstack_copy0 (&result->mem_pool,
					    repfile->token.val.str.startmb,
					    repfile->token.val.str.lenmb);

	  state = 5;
	  continue;

	case 90:
	  if (nowtok != tok_charids)
	    lr_error (repfile, _("\
%1$s: definition does not end with `END %1$s'"), "CHARIDS");

	  lr_ignore_rest (repfile, nowtok == tok_charids);
	  break;
	}

      break;
    }

  if (state != 2 && state != 90 && !be_quiet)
    record_error (0, 0, _("%s: premature end of file"),
		  repfile->fname);

  lr_close (repfile);

  if (tsearch (result, &known, &repertoire_compare) == NULL)
    /* Something went wrong.  */
    record_error (0, errno, _("cannot save new repertoire map"));

  return result;
}


void
repertoire_complain (const char *name)
{
  if (tfind (name, &unavailable, (__compar_fn_t) strcmp) == NULL)
    {
      record_error (0, errno, _("\
repertoire map file `%s' not found"), name);

      /* Remember that we reported this map.  */
      tsearch (name, &unavailable, (__compar_fn_t) strcmp);
    }
}


static int
repertoire_compare (const void *p1, const void *p2)
{
  struct repertoire_t *r1 = (struct repertoire_t *) p1;
  struct repertoire_t *r2 = (struct repertoire_t *) p2;

  return strcmp (r1->name, r2->name);
}


static const struct keyword_t *
repertoiremap_hash (const char *str, size_t len)
{
  static const struct keyword_t wordlist[] =
  {
    {"escape_char",      tok_escape_char,     0},
    {"comment_char",     tok_comment_char,    0},
    {"CHARIDS",          tok_charids,         0},
    {"END",              tok_end,             0},
  };

  if (len == 11 && memcmp (wordlist[0].name, str, 11) == 0)
    return &wordlist[0];
  if (len == 12 && memcmp (wordlist[1].name, str, 12) == 0)
    return &wordlist[1];
  if (len == 7 && memcmp (wordlist[2].name, str, 7) == 0)
    return &wordlist[2];
  if (len == 3 && memcmp (wordlist[3].name, str, 3) == 0)
    return &wordlist[3];

  return NULL;
}


static void
repertoire_new_char (struct linereader *lr, hash_table *ht, hash_table *rt,
		     struct obstack *ob, uint32_t value, const char *from,
		     const char *to, int decimal_ellipsis)
{
  char *from_end;
  char *to_end;
  const char *cp;
  char *buf = NULL;
  int prefix_len, len1, len2;
  unsigned long int from_nr, to_nr, cnt;

  if (to == NULL)
    {
      insert_entry (ht, from, strlen (from),
		    (void *) (unsigned long int) value);
      /* Please note that it isn't a bug if a symbol is defined more
	 than once.  All later definitions are simply discarded.  */

      insert_entry (rt, obstack_copy (ob, &value, sizeof (value)),
		    sizeof (value), (void *) from);

      return;
    }

  /* We have a range: the names must have names with equal prefixes
     and an equal number of digits, where the second number is greater
     or equal than the first.  */
  len1 = strlen (from);
  len2 = strlen (to);

  if (len1 != len2)
    {
    invalid_range:
      lr_error (lr, _("invalid names for character range"));
      return;
    }

  cp = &from[len1 - 1];
  if (decimal_ellipsis)
    while (isdigit (*cp) && cp >= from)
      --cp;
  else
    while (isxdigit (*cp) && cp >= from)
      {
	if (!isdigit (*cp) && !isupper (*cp))
	  lr_error (lr, _("\
hexadecimal range format should use only capital characters"));
	--cp;
      }

  prefix_len = (cp - from) + 1;

  if (cp == &from[len1 - 1] || strncmp (from, to, prefix_len) != 0)
    goto invalid_range;

  errno = 0;
  from_nr = strtoul (&from[prefix_len], &from_end, decimal_ellipsis ? 10 : 16);
  if (*from_end != '\0' || (from_nr == ULONG_MAX && errno == ERANGE)
      || ((to_nr = strtoul (&to[prefix_len], &to_end,
			    decimal_ellipsis ? 10 : 16)) == ULONG_MAX
          && errno == ERANGE)
      || *to_end != '\0')
    {
      lr_error (lr, _("<%s> and <%s> are invalid names for range"),
		from, to);
      return;
    }

  if (from_nr > to_nr)
    {
      lr_error (lr, _("upper limit in range is smaller than lower limit"));
      return;
    }

  for (cnt = from_nr; cnt <= to_nr; ++cnt)
    {
      uint32_t this_value = value + (cnt - from_nr);

      obstack_printf (ob, decimal_ellipsis ? "%.*s%0*ld" : "%.*s%0*lX",
		      prefix_len, from, len1 - prefix_len, cnt);
      obstack_1grow (ob, '\0');

      insert_entry (ht, buf, len1,
		    (void *) (unsigned long int) this_value);
      /* Please note we don't examine the return value since it is no error
	 if we have two definitions for a symbol.  */

      insert_entry (rt, obstack_copy (ob, &this_value, sizeof (this_value)),
		    sizeof (this_value), (void *) from);
    }
}


uint32_t
repertoire_find_value (const struct repertoire_t *rep, const char *name,
		       size_t len)
{
  void *result;

  if (rep == NULL)
    return ILLEGAL_CHAR_VALUE;

  if (find_entry ((hash_table *) &rep->char_table, name, len, &result) < 0)
    return ILLEGAL_CHAR_VALUE;

  return (uint32_t) ((unsigned long int) result);
}


const char *
repertoire_find_symbol (const struct repertoire_t *rep, uint32_t ucs)
{
  void *result;

  if (rep == NULL)
    return NULL;

  if (find_entry ((hash_table *) &rep->reverse_table, &ucs, sizeof (ucs),
		  &result) < 0)
    return NULL;

  return (const char *) result;
}


struct charseq *
repertoire_find_seq (const struct repertoire_t *rep, uint32_t ucs)
{
  void *result;

  if (rep == NULL)
    return NULL;

  if (find_entry ((hash_table *) &rep->seq_table, &ucs, sizeof (ucs),
		  &result) < 0)
    return NULL;

  return (struct charseq *) result;
}
