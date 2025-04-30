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
#include <string.h>
#include <stdint.h>
#include <sys/uio.h>

#include <assert.h>

#include "localedef.h"
#include "localeinfo.h"
#include "locfile.h"


/* The real definition of the struct for the LC_NAME locale.  */
struct locale_name_t
{
  const char *name_fmt;
  const char *name_gen;
  const char *name_mr;
  const char *name_mrs;
  const char *name_miss;
  const char *name_ms;
};


static void
name_startup (struct linereader *lr, struct localedef_t *locale,
	       int ignore_content)
{
  if (!ignore_content)
    locale->categories[LC_NAME].name =
      (struct locale_name_t *) xcalloc (1, sizeof (struct locale_name_t));

  if (lr != NULL)
    {
      lr->translate_strings = 1;
      lr->return_widestr = 0;
    }
}


void
name_finish (struct localedef_t *locale, const struct charmap_t *charmap)
{
  struct locale_name_t *name = locale->categories[LC_NAME].name;
  int nothing = 0;

  /* Now resolve copying and also handle completely missing definitions.  */
  if (name == NULL)
    {
      /* First see whether we were supposed to copy.  If yes, find the
	 actual definition.  */
      if (locale->copy_name[LC_NAME] != NULL)
	{
	  /* Find the copying locale.  This has to happen transitively since
	     the locale we are copying from might also copying another one.  */
	  struct localedef_t *from = locale;

	  do
	    from = find_locale (LC_NAME, from->copy_name[LC_NAME],
				from->repertoire_name, charmap);
	  while (from->categories[LC_NAME].name == NULL
		 && from->copy_name[LC_NAME] != NULL);

	  name = locale->categories[LC_NAME].name
	    = from->categories[LC_NAME].name;
	}

      /* If there is still no definition issue an warning and create an
	 empty one.  */
      if (name == NULL)
	{
	  record_warning (_("\
No definition for %s category found"), "LC_NAME");
	  name_startup (NULL, locale, 0);
	  name = locale->categories[LC_NAME].name;
	  nothing = 1;
	}
    }

  if (name->name_fmt == NULL)
    {
      if (! nothing)
	record_error (0, 0, _("%s: field `%s' not defined"),
		      "LC_NAME", "name_fmt");
      /* Use as the default value the value of the i18n locale.  */
      name->name_fmt = "%p%t%g%t%m%t%f";
    }
  else
    {
      /* We must check whether the format string contains only the
	 allowed escape sequences.  */
      const char *cp = name->name_fmt;

      if (*cp == '\0')
	record_error (0, 0, _("%s: field `%s' must not be empty"),
		      "LC_NAME", "name_fmt");
      else
	while (*cp != '\0')
	  {
	    if (*cp == '%')
	      {
		if (*++cp == 'R')
		  /* Romanize-flag.  */
		  ++cp;
		if (strchr ("dfFgGlomMpsSt", *cp) == NULL)
		  {
		    record_error (0, 0, _("\
%s: invalid escape sequence in field `%s'"), "LC_NAME", "name_fmt");
		    break;
		  }
	      }
	    ++cp;
	  }
    }

#define TEST_ELEM(cat) \
  if (name->cat == NULL)						      \
    {									      \
      if (verbose && ! nothing)						      \
	record_warning (_("%s: field `%s' not defined"), "LC_NAME", #cat);    \
      name->cat = "";							      \
    }

  TEST_ELEM (name_gen);
  TEST_ELEM (name_mr);
  TEST_ELEM (name_mrs);
  TEST_ELEM (name_miss);
  TEST_ELEM (name_ms);
}


void
name_output (struct localedef_t *locale, const struct charmap_t *charmap,
	     const char *output_path)
{
  struct locale_name_t *name = locale->categories[LC_NAME].name;
  struct locale_file file;

  init_locale_data (&file, _NL_ITEM_INDEX (_NL_NUM_LC_NAME));
  add_locale_string (&file, name->name_fmt);
  add_locale_string (&file, name->name_gen);
  add_locale_string (&file, name->name_mr);
  add_locale_string (&file, name->name_mrs);
  add_locale_string (&file, name->name_miss);
  add_locale_string (&file, name->name_ms);
  add_locale_string (&file, charmap->code_set_name);
  write_locale_data (output_path, LC_NAME, "LC_NAME", &file);
}


/* The parser for the LC_NAME section of the locale definition.  */
void
name_read (struct linereader *ldfile, struct localedef_t *result,
	   const struct charmap_t *charmap, const char *repertoire_name,
	   int ignore_content)
{
  struct locale_name_t *name;
  struct token *now;
  struct token *arg;
  enum token_t nowtok;

  /* The rest of the line containing `LC_NAME' must be empty.  */
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
      handle_copy (ldfile, charmap, repertoire_name, result, tok_lc_name,
		   LC_NAME, "LC_NAME", ignore_content);
      return;
    }

  /* Prepare the data structures.  */
  name_startup (ldfile, result, ignore_content);
  name = result->categories[LC_NAME].name;

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
	  if (name->cat != NULL)					      \
	    lr_error (ldfile, _("%s: field `%s' declared more than once"),    \
		      "LC_NAME", #cat);					      \
	  else if (!ignore_content && arg->val.str.startmb == NULL)	      \
	    {								      \
	      lr_error (ldfile, _("%s: unknown character in field `%s'"),     \
			"LC_NAME", #cat);				      \
	      name->cat = "";						      \
	    }								      \
	  else if (!ignore_content)					      \
	    name->cat = arg->val.str.startmb;				      \
	  break

	  STR_ELEM (name_fmt);
	  STR_ELEM (name_gen);
	  STR_ELEM (name_mr);
	  STR_ELEM (name_mrs);
	  STR_ELEM (name_miss);
	  STR_ELEM (name_ms);

	case tok_end:
	  /* Next we assume `LC_NAME'.  */
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok == tok_eof)
	    break;
	  if (arg->tok == tok_eol)
	    lr_error (ldfile, _("%s: incomplete `END' line"), "LC_NAME");
	  else if (arg->tok != tok_lc_name)
	    lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_NAME");
	  lr_ignore_rest (ldfile, arg->tok == tok_lc_name);
	  return;

	default:
	err_label:
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_NAME");
	}

      /* Prepare for the next round.  */
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }

  /* When we come here we reached the end of the file.  */
  lr_error (ldfile, _("%s: premature end of file"), "LC_NAME");
}
