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

#include <langinfo.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/uio.h>

#include <assert.h>

#include "localedef.h"
#include "localeinfo.h"
#include "locfile.h"


/* The real definition of the struct for the LC_IDENTIFICATION locale.  */
struct locale_identification_t
{
  const char *title;
  const char *source;
  const char *address;
  const char *contact;
  const char *email;
  const char *tel;
  const char *fax;
  const char *language;
  const char *territory;
  const char *audience;
  const char *application;
  const char *abbreviation;
  const char *revision;
  const char *date;
  const char *category[__LC_LAST];
};


static const char *category_name[__LC_LAST] =
{
  [LC_CTYPE] = "LC_CTYPE",
  [LC_NUMERIC] = "LC_NUMERIC",
  [LC_TIME] = "LC_TIME",
  [LC_COLLATE] = "LC_COLLATE",
  [LC_MONETARY] = "LC_MONETARY",
  [LC_MESSAGES] = "LC_MESSAGES",
  [LC_ALL] = "LC_ALL",
  [LC_PAPER] = "LC_PAPER",
  [LC_NAME] = "LC_NAME",
  [LC_ADDRESS] = "LC_ADDRESS",
  [LC_TELEPHONE] = "LC_TELEPHONE",
  [LC_MEASUREMENT] = "LC_MEASUREMENT",
  [LC_IDENTIFICATION] = "LC_IDENTIFICATION"
};


static void
identification_startup (struct linereader *lr, struct localedef_t *locale,
			int ignore_content)
{
  if (!ignore_content)
    {
      locale->categories[LC_IDENTIFICATION].identification =
	(struct locale_identification_t *)
	xcalloc (1, sizeof (struct locale_identification_t));

      locale->categories[LC_IDENTIFICATION].identification->category[LC_ALL] =
	"";
    }

  if (lr != NULL)
    {
      lr->translate_strings = 1;
      lr->return_widestr = 0;
    }
}


void
identification_finish (struct localedef_t *locale,
		       const struct charmap_t *charmap)
{
  struct locale_identification_t *identification
    = locale->categories[LC_IDENTIFICATION].identification;
  int nothing = 0;
  size_t num;

  /* Now resolve copying and also handle completely missing definitions.  */
  if (identification == NULL)
    {
      /* First see whether we were supposed to copy.  If yes, find the
	 actual definition.  */
      if (locale->copy_name[LC_IDENTIFICATION] != NULL)
	{
	  /* Find the copying locale.  This has to happen transitively since
	     the locale we are copying from might also copying another one.  */
	  struct localedef_t *from = locale;

	  do
	    from = find_locale (LC_IDENTIFICATION,
				from->copy_name[LC_IDENTIFICATION],
				from->repertoire_name, charmap);
	  while (from->categories[LC_IDENTIFICATION].identification == NULL
		 && from->copy_name[LC_IDENTIFICATION] != NULL);

	  identification = locale->categories[LC_IDENTIFICATION].identification
	    = from->categories[LC_IDENTIFICATION].identification;
	}

      /* If there is still no definition issue an warning and create an
	 empty one.  */
      if (identification == NULL)
	{
	  record_warning (_("\
No definition for %s category found"), "LC_IDENTIFICATION");
	  identification_startup (NULL, locale, 0);
	  identification
	    = locale->categories[LC_IDENTIFICATION].identification;
	  nothing = 1;
	}
    }

#define TEST_ELEM(cat) \
  if (identification->cat == NULL)					      \
    {									      \
      if (verbose && ! nothing)						      \
	record_warning (_("%s: field `%s' not defined"), "LC_IDENTIFICATION", \
			#cat);						      \
      identification->cat = "";						      \
    }

  TEST_ELEM (title);
  TEST_ELEM (source);
  TEST_ELEM (address);
  TEST_ELEM (contact);
  TEST_ELEM (email);
  TEST_ELEM (tel);
  TEST_ELEM (fax);
  TEST_ELEM (language);
  TEST_ELEM (territory);
  TEST_ELEM (audience);
  TEST_ELEM (application);
  TEST_ELEM (abbreviation);
  TEST_ELEM (revision);
  TEST_ELEM (date);

  for (num = 0; num < __LC_LAST; ++num)
    {
      /* We don't accept/parse this category, so skip it early.  */
      if (num == LC_ALL)
	continue;

      if (identification->category[num] == NULL)
	{
	  if (verbose && ! nothing)
	    record_warning (_("\
%s: no identification for category `%s'"), "LC_IDENTIFICATION",
			    category_name[num]);
	  identification->category[num] = "";
	}
      else
	{
	  /* Only list the standards we care about.  This is based on the
	     ISO 30112 WD10 [2014] standard which supersedes all previous
	     revisions of the ISO 14652 standard.  */
	  static const char * const standards[] =
	    {
	      "posix:1993",
	      "i18n:2004",
	      "i18n:2012",
	    };
	  size_t i;
	  bool matched = false;

	  for (i = 0; i < sizeof (standards) / sizeof (standards[0]); ++i)
	    if (strcmp (identification->category[num], standards[i]) == 0)
	      matched = true;

	  if (matched != true)
	    record_error (0, 0, _("\
%s: unknown standard `%s' for category `%s'"),
			  "LC_IDENTIFICATION",
			  identification->category[num],
			  category_name[num]);
	}
    }
}


void
identification_output (struct localedef_t *locale,
		       const struct charmap_t *charmap,
		       const char *output_path)
{
  struct locale_identification_t *identification
    = locale->categories[LC_IDENTIFICATION].identification;
  struct locale_file file;
  size_t num;

  init_locale_data (&file, _NL_ITEM_INDEX (_NL_NUM_LC_IDENTIFICATION));
  add_locale_string (&file, identification->title);
  add_locale_string (&file, identification->source);
  add_locale_string (&file, identification->address);
  add_locale_string (&file, identification->contact);
  add_locale_string (&file, identification->email);
  add_locale_string (&file, identification->tel);
  add_locale_string (&file, identification->fax);
  add_locale_string (&file, identification->language);
  add_locale_string (&file, identification->territory);
  add_locale_string (&file, identification->audience);
  add_locale_string (&file, identification->application);
  add_locale_string (&file, identification->abbreviation);
  add_locale_string (&file, identification->revision);
  add_locale_string (&file, identification->date);
  start_locale_structure (&file);
  for (num = 0; num < __LC_LAST; ++num)
    if (num != LC_ALL)
      add_locale_string (&file, identification->category[num]);
  end_locale_structure (&file);
  add_locale_string (&file, charmap->code_set_name);
  write_locale_data (output_path, LC_IDENTIFICATION, "LC_IDENTIFICATION",
		     &file);
}


/* The parser for the LC_IDENTIFICATION section of the locale definition.  */
void
identification_read (struct linereader *ldfile, struct localedef_t *result,
	       const struct charmap_t *charmap, const char *repertoire_name,
	       int ignore_content)
{
  struct locale_identification_t *identification;
  struct token *now;
  struct token *arg;
  struct token *cattok;
  int category;
  enum token_t nowtok;

  /* The rest of the line containing `LC_IDENTIFICATION' must be free.  */
  lr_ignore_rest (ldfile, 1);

  do
    {
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }
  while (nowtok == tok_eol);

  /* If we see `copy' now we are almost done.  */
  if (nowtok == tok_copy)
    {
      handle_copy (ldfile, charmap, repertoire_name, result,
		   tok_lc_identification, LC_IDENTIFICATION,
		   "LC_IDENTIFICATION", ignore_content);
      return;
    }

  /* Prepare the data structures.  */
  identification_startup (ldfile, result, ignore_content);
  identification = result->categories[LC_IDENTIFICATION].identification;

  while (1)
    {
      /* Of course we don't proceed beyond the end of file.  */
      if (nowtok == tok_eof)
	break;

      /* Ignore empty lines.  */
      if (nowtok == tok_eol)
	{
	  now = lr_token (ldfile, charmap, result, NULL, verbose);
	  nowtok = now->tok;
	  continue;
	}

      switch (nowtok)
	{
#define STR_ELEM(cat) \
	case tok_##cat:							      \
	  /* Ignore the rest of the line if we don't need the input of	      \
	     this line.  */						      \
	  if (ignore_content)						      \
	    {								      \
	      lr_ignore_rest (ldfile, 0);				      \
	      break;							      \
	    }								      \
									      \
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);	      \
	  if (arg->tok != tok_string)					      \
	    goto err_label;						      \
	  if (identification->cat != NULL)				      \
	    lr_error (ldfile, _("\
%s: field `%s' declared more than once"), "LC_IDENTIFICATION", #cat);	      \
	  else if (!ignore_content && arg->val.str.startmb == NULL)	      \
	    {								      \
	      lr_error (ldfile, _("\
%s: unknown character in field `%s'"), "LC_IDENTIFICATION", #cat);	      \
	      identification->cat = "";					      \
	    }								      \
	  else if (!ignore_content)					      \
	    identification->cat = arg->val.str.startmb;			      \
	  break

	  STR_ELEM (title);
	  STR_ELEM (source);
	  STR_ELEM (address);
	  STR_ELEM (contact);
	  STR_ELEM (email);
	  STR_ELEM (tel);
	  STR_ELEM (fax);
	  STR_ELEM (language);
	  STR_ELEM (territory);
	  STR_ELEM (audience);
	  STR_ELEM (application);
	  STR_ELEM (abbreviation);
	  STR_ELEM (revision);
	  STR_ELEM (date);

	case tok_category:
	  /* Ignore the rest of the line if we don't need the input of
	     this line.  */
	  if (ignore_content)
	    {
	      lr_ignore_rest (ldfile, 0);
	      break;
	    }

	  /* We expect two operands.  */
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok != tok_string && arg->tok != tok_ident)
	    goto err_label;
	  /* Next is a semicolon.  */
	  cattok = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (cattok->tok != tok_semicolon)
	    goto err_label;
	  /* Now a LC_xxx identifier.  */
	  cattok = lr_token (ldfile, charmap, result, NULL, verbose);
	  switch (cattok->tok)
	    {
#define CATEGORY(lname, uname) \
	    case tok_lc_##lname:					      \
	      category = LC_##uname;					      \
	      break

	      CATEGORY (identification, IDENTIFICATION);
	      CATEGORY (ctype, CTYPE);
	      CATEGORY (collate, COLLATE);
	      CATEGORY (time, TIME);
	      CATEGORY (numeric, NUMERIC);
	      CATEGORY (monetary, MONETARY);
	      CATEGORY (messages, MESSAGES);
	      CATEGORY (paper, PAPER);
	      CATEGORY (name, NAME);
	      CATEGORY (address, ADDRESS);
	      CATEGORY (telephone, TELEPHONE);
	      CATEGORY (measurement, MEASUREMENT);

	    default:
	      goto err_label;
	    }
	  if (identification->category[category] != NULL)
	    {
	      lr_error (ldfile, _("\
%s: duplicate category version definition"), "LC_IDENTIFICATION");
	      free (arg->val.str.startmb);
	    }
	  else
	    identification->category[category] = arg->val.str.startmb;
	  break;

	case tok_end:
	  /* Next we assume `LC_IDENTIFICATION'.  */
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok == tok_eof)
	    break;
	  if (arg->tok == tok_eol)
	    lr_error (ldfile, _("%s: incomplete `END' line"),
		      "LC_IDENTIFICATION");
	  else if (arg->tok != tok_lc_identification)
	    lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_IDENTIFICATION");
	  lr_ignore_rest (ldfile, arg->tok == tok_lc_identification);
	  return;

	default:
	err_label:
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_IDENTIFICATION");
	}

      /* Prepare for the next round.  */
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }

  /* When we come here we reached the end of the file.  */
  lr_error (ldfile, _("%s: premature end of file"), "LC_IDENTIFICATION");
}
