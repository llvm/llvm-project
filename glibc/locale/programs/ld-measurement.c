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


/* The real definition of the struct for the LC_MEASUREMENT locale.  */
struct locale_measurement_t
{
  unsigned char measurement;
};


static void
measurement_startup (struct linereader *lr, struct localedef_t *locale,
		     int ignore_content)
{
  if (!ignore_content)
    locale->categories[LC_MEASUREMENT].measurement =
      (struct locale_measurement_t *)
      xcalloc (1, sizeof (struct locale_measurement_t));

  if (lr != NULL)
    {
      lr->translate_strings = 1;
      lr->return_widestr = 0;
    }
}


void
measurement_finish (struct localedef_t *locale,
		    const struct charmap_t *charmap)
{
  struct locale_measurement_t *measurement =
    locale->categories[LC_MEASUREMENT].measurement;
  int nothing = 0;

  /* Now resolve copying and also handle completely missing definitions.  */
  if (measurement == NULL)
    {
      /* First see whether we were supposed to copy.  If yes, find the
	 actual definition.  */
      if (locale->copy_name[LC_MEASUREMENT] != NULL)
	{
	  /* Find the copying locale.  This has to happen transitively since
	     the locale we are copying from might also copying another one.  */
	  struct localedef_t *from = locale;

	  do
	    from = find_locale (LC_MEASUREMENT,
				from->copy_name[LC_MEASUREMENT],
				from->repertoire_name, charmap);
	  while (from->categories[LC_MEASUREMENT].measurement == NULL
		 && from->copy_name[LC_MEASUREMENT] != NULL);

	  measurement = locale->categories[LC_MEASUREMENT].measurement
	    = from->categories[LC_MEASUREMENT].measurement;
	}

      /* If there is still no definition issue an warning and create an
	 empty one.  */
      if (measurement == NULL)
	{
	  record_warning (_("\
No definition for %s category found"), "LC_MEASUREMENT");
	  measurement_startup (NULL, locale, 0);
	  measurement = locale->categories[LC_MEASUREMENT].measurement;
	  nothing = 1;
	}
    }

  if (measurement->measurement == 0)
    {
      if (! nothing)
	record_error (0, 0, _("%s: field `%s' not defined"),
		      "LC_MEASUREMENT", "measurement");
      /* Use as the default value the value of the i18n locale.  */
      measurement->measurement = 1;
    }
  else
    {
      if (measurement->measurement > 3)
	record_error (0, 0, _("%s: invalid value for field `%s'"),
		      "LC_MEASUREMENT", "measurement");
    }
}


void
measurement_output (struct localedef_t *locale,
		    const struct charmap_t *charmap, const char *output_path)
{
  struct locale_measurement_t *measurement =
    locale->categories[LC_MEASUREMENT].measurement;
  struct locale_file file;

  init_locale_data (&file, _NL_ITEM_INDEX (_NL_NUM_LC_MEASUREMENT));
  add_locale_char (&file, measurement->measurement);
  add_locale_string (&file, charmap->code_set_name);
  write_locale_data (output_path, LC_MEASUREMENT, "LC_MEASUREMENT", &file);
}


/* The parser for the LC_MEASUREMENT section of the locale definition.  */
void
measurement_read (struct linereader *ldfile, struct localedef_t *result,
		  const struct charmap_t *charmap, const char *repertoire_name,
		  int ignore_content)
{
  struct locale_measurement_t *measurement;
  struct token *now;
  struct token *arg;
  enum token_t nowtok;

  /* The rest of the line containing `LC_MEASUREMENT' must be free.  */
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
		   tok_lc_measurement, LC_MEASUREMENT, "LC_MEASUREMENT",
		   ignore_content);
      return;
    }

  /* Prepare the data structures.  */
  measurement_startup (ldfile, result, ignore_content);
  measurement = result->categories[LC_MEASUREMENT].measurement;

  while (1)
    {
      /* Of course we don't proceed beyond the end of file.  */
      if (nowtok == tok_eof)
	break;

      /* Ingore empty lines.  */
      if (nowtok == tok_eol)
	{
	  now = lr_token (ldfile, charmap, result, NULL, verbose);
	  nowtok = now->tok;
	  continue;
	}

      switch (nowtok)
	{
#define INT_ELEM(cat) \
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
	  if (arg->tok != tok_number)					      \
	    goto err_label;						      \
	  else if (measurement->cat != 0)				      \
	    lr_error (ldfile, _("%s: field `%s' declared more than once"),    \
		      "LC_MEASUREMENT", #cat);				      \
	  else if (!ignore_content)					      \
	    measurement->cat = arg->val.num;				      \
	  break

	  INT_ELEM (measurement);

	case tok_end:
	  /* Next we assume `LC_MEASUREMENT'.  */
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok == tok_eof)
	    break;
	  if (arg->tok == tok_eol)
	    lr_error (ldfile, _("%s: incomplete `END' line"),
		      "LC_MEASUREMENT");
	  else if (arg->tok != tok_lc_measurement)
	    lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_MEASUREMENT");
	  lr_ignore_rest (ldfile, arg->tok == tok_lc_measurement);
	  return;

	default:
	err_label:
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_MEASUREMENT");
	}

      /* Prepare for the next round.  */
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }

  /* When we come here we reached the end of the file.  */
  lr_error (ldfile, _("%s: premature end of file"),
	    "LC_MEASUREMENT");
}
