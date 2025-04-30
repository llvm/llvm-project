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

#include <byteswap.h>
#include <langinfo.h>
#include <string.h>
#include <stdint.h>
#include <sys/uio.h>

#include <assert.h>

#include "localedef.h"
#include "localeinfo.h"
#include "locfile.h"


static struct
{
  const char ab2[3];
  const char ab3[4];
  uint32_t num;
} iso3166[] =
{
#define DEFINE_COUNTRY_CODE(Name, Ab2, Ab3, Num) \
  { #Ab2, #Ab3, Num },
#include "iso-3166.def"
};


static struct
{
  const char ab[3];
  const char term[4];
  const char lib[4];
} iso639[] =
{
#define DEFINE_LANGUAGE_CODE(Name, Ab, Term, Lib) \
  { #Ab, #Term, #Lib },
#define DEFINE_LANGUAGE_CODE3(Name, Term, Lib) \
  { "", #Term, #Lib },
#define DEFINE_LANGUAGE_CODE2(Name, Term) \
  { "", #Term, "" },
#include "iso-639.def"
};


/* The real definition of the struct for the LC_ADDRESS locale.  */
struct locale_address_t
{
  const char *postal_fmt;
  const char *country_name;
  const char *country_post;
  const char *country_ab2;
  const char *country_ab3;
  uint32_t country_num;
  const char *country_car;
  const char *country_isbn;
  const char *lang_name;
  const char *lang_ab;
  const char *lang_term;
  const char *lang_lib;
};


static void
address_startup (struct linereader *lr, struct localedef_t *locale,
		 int ignore_content)
{
  if (!ignore_content)
    locale->categories[LC_ADDRESS].address =
      (struct locale_address_t *) xcalloc (1,
					   sizeof (struct locale_address_t));

  if (lr != NULL)
    {
      lr->translate_strings = 1;
      lr->return_widestr = 0;
    }
}


void
address_finish (struct localedef_t *locale, const struct charmap_t *charmap)
{
  struct locale_address_t *address = locale->categories[LC_ADDRESS].address;
  size_t cnt;
  int helper;
  int nothing = 0;

  /* Now resolve copying and also handle completely missing definitions.  */
  if (address == NULL)
    {
      /* First see whether we were supposed to copy.  If yes, find the
	 actual definition.  */
      if (locale->copy_name[LC_ADDRESS] != NULL)
	{
	  /* Find the copying locale.  This has to happen transitively since
	     the locale we are copying from might also copying another one.  */
	  struct localedef_t *from = locale;

	  do
	    from = find_locale (LC_ADDRESS, from->copy_name[LC_ADDRESS],
				from->repertoire_name, charmap);
	  while (from->categories[LC_ADDRESS].address == NULL
		 && from->copy_name[LC_ADDRESS] != NULL);

	  address = locale->categories[LC_ADDRESS].address
	    = from->categories[LC_ADDRESS].address;
	}

      /* If there is still no definition issue an warning and create an
	 empty one.  */
      if (address == NULL)
	{
	  record_warning (_("\
No definition for %s category found"), "LC_ADDRESS");
	  address_startup (NULL, locale, 0);
	  address = locale->categories[LC_ADDRESS].address;
	  nothing = 1;
	}
    }

  if (address->postal_fmt == NULL)
    {
      if (! nothing)
	record_error (0, 0, _("%s: field `%s' not defined"),
		      "LC_ADDRESS", "postal_fmt");
      /* Use as the default value the value of the i18n locale.  */
      address->postal_fmt = "%a%N%f%N%d%N%b%N%s %h %e %r%N%C-%z %T%N%c%N";
    }
  else
    {
      /* We must check whether the format string contains only the allowed
	 escape sequences.  Last checked against ISO 30112 WD10 [2014]. */
      const char *cp = address->postal_fmt;

      if (*cp == '\0')
	record_error (0, 0, _("%s: field `%s' must not be empty"),
		      "LC_ADDRESS", "postal_fmt");
      else
	while (*cp != '\0')
	  {
	    if (*cp == '%')
	      {
		if (*++cp == 'R')
		  /* Romanize-flag.  */
		  ++cp;
		if (strchr ("nafdbshNtreClzTSc%", *cp) == NULL)
		  {
		    record_error (0, 0, _("\
%s: invalid escape `%%%c' sequence in field `%s'"),
				  "LC_ADDRESS", *cp, "postal_fmt");
		    break;
		  }
	      }
	    ++cp;
	  }
    }

#define TEST_ELEM(cat) \
  if (address->cat == NULL)						      \
    {									      \
      if (verbose && ! nothing)						      \
	record_warning (_("%s: field `%s' not defined"), "LC_ADDRESS", #cat); \
      address->cat = "";						      \
    }

  TEST_ELEM (country_name);
  /* XXX Test against list of defined codes.  */
  TEST_ELEM (country_post);
  /* XXX Test against list of defined codes.  */
  TEST_ELEM (country_car);
  /* XXX Test against list of defined codes.  */
  TEST_ELEM (country_isbn);
  TEST_ELEM (lang_name);

  helper = 1;
  if (address->lang_term == NULL)
    {
      if (verbose && ! nothing)
	record_warning (_("%s: field `%s' not defined"), "LC_ADDRESS",
			"lang_term");
      address->lang_term = "";
      cnt = sizeof (iso639) / sizeof (iso639[0]);
    }
  else if (address->lang_term[0] == '\0')
    {
      if (verbose)
	record_warning (_("%s: field `%s' must not be empty"), "LC_ADDRESS",
			"lang_term");
      cnt = sizeof (iso639) / sizeof (iso639[0]);
    }
  else
    {
      /* Look for this language in the table.  */
      for (cnt = 0; cnt < sizeof (iso639) / sizeof (iso639[0]); ++cnt)
	if (strcmp (address->lang_term, iso639[cnt].term) == 0)
	  break;
      if (cnt == sizeof (iso639) / sizeof (iso639[0]))
	record_error (0, 0, _("\
%s: terminology language code `%s' not defined"),
		      "LC_ADDRESS", address->lang_term);
    }

  if (address->lang_ab == NULL)
    {
      if ((cnt == sizeof (iso639) / sizeof (iso639[0])
	   || iso639[cnt].ab[0] != '\0')
	  && verbose && ! nothing)
	record_warning (_("%s: field `%s' not defined"), "LC_ADDRESS",
			"lang_ab");
      address->lang_ab = "";
    }
  else if (address->lang_ab[0] == '\0')
    {
      if ((cnt == sizeof (iso639) / sizeof (iso639[0])
	   || iso639[cnt].ab[0] != '\0')
	  && verbose)
	record_warning (_("%s: field `%s' must not be empty"),
			"LC_ADDRESS", "lang_ab");
    }
  else if (cnt < sizeof (iso639) / sizeof (iso639[0])
	   && iso639[cnt].ab[0] == '\0')
    {
      record_error (0, 0, _("%s: field `%s' must not be defined"),
		    "LC_ADDRESS", "lang_ab");

      address->lang_ab = "";
    }
  else
    {
      if (cnt == sizeof (iso639) / sizeof (iso639[0]))
	{
	  helper = 2;
	  for (cnt = 0; cnt < sizeof (iso639) / sizeof (iso639[0]); ++cnt)
	    if (strcmp (address->lang_ab, iso639[cnt].ab) == 0)
	      break;
	  if (cnt == sizeof (iso639) / sizeof (iso639[0]))
	    record_error (0, 0, _("\
%s: language abbreviation `%s' not defined"),
			  "LC_ADDRESS", address->lang_ab);
	}
      else
	if (strcmp (iso639[cnt].ab, address->lang_ab) != 0
	    && iso639[cnt].ab[0] != '\0')
	  record_error (0, 0, _("\
%s: `%s' value does not match `%s' value"),
			"LC_ADDRESS", "lang_ab", "lang_term");
    }

  if (address->lang_lib == NULL)
    /* This is no error.  */
    address->lang_lib = address->lang_term;
  else if (address->lang_lib[0] == '\0')
    {
      if (verbose)
	record_warning (_("%s: field `%s' must not be empty"),
			"LC_ADDRESS", "lang_lib");
    }
  else
    {
      if (cnt == sizeof (iso639) / sizeof (iso639[0]))
	{
	  for (cnt = 0; cnt < sizeof (iso639) / sizeof (iso639[0]); ++cnt)
	    if (strcmp (address->lang_lib, iso639[cnt].lib) == 0)
	      break;
	  if (cnt == sizeof (iso639) / sizeof (iso639[0]))
	    record_error (0, 0, _("\
%s: language abbreviation `%s' not defined"),
			  "LC_ADDRESS", address->lang_lib);
	}
      else
	if (strcmp (iso639[cnt].ab, address->lang_ab) != 0)
	  record_error (0, 0, _("\
%s: `%s' value does not match `%s' value"), "LC_ADDRESS", "lang_lib",
			helper == 1 ? "lang_term" : "lang_ab");
    }

  if (address->country_num == 0)
    {
      if (verbose && ! nothing)
	record_warning (_("%s: field `%s' not defined"), "LC_ADDRESS",
			"country_num");
      cnt = sizeof (iso3166) / sizeof (iso3166[0]);
    }
  else
    {
      for (cnt = 0; cnt < sizeof (iso3166) / sizeof (iso3166[0]); ++cnt)
	if (address->country_num == iso3166[cnt].num)
	  break;

      if (cnt == sizeof (iso3166) / sizeof (iso3166[0]))
	record_error (0, 0, _("\
%s: numeric country code `%d' not valid"),
		      "LC_ADDRESS", address->country_num);
    }

  if (address->country_ab2 == NULL)
    {
      if (verbose && ! nothing)
	record_warning (_("%s: field `%s' not defined"), "LC_ADDRESS",
			"country_ab2");
      address->country_ab2 = "  ";
    }
  else if (cnt != sizeof (iso3166) / sizeof (iso3166[0])
	   && strcmp (address->country_ab2, iso3166[cnt].ab2) != 0)
    record_error (0, 0, _("%s: `%s' value does not match `%s' value"),
		  "LC_ADDRESS", "country_ab2", "country_num");

  if (address->country_ab3 == NULL)
    {
      if (verbose && ! nothing)
	record_warning (_("%s: field `%s' not defined"), "LC_ADDRESS",
			"country_ab3");
      address->country_ab3 = "   ";
    }
  else if (cnt != sizeof (iso3166) / sizeof (iso3166[0])
	   && strcmp (address->country_ab3, iso3166[cnt].ab3) != 0)
    record_error (0, 0, _("\
%s: `%s' value does not match `%s' value"),
		  "LC_ADDRESS", "country_ab3", "country_num");
}


void
address_output (struct localedef_t *locale, const struct charmap_t *charmap,
		const char *output_path)
{
  struct locale_address_t *address = locale->categories[LC_ADDRESS].address;
  struct locale_file file;

  init_locale_data (&file, _NL_ITEM_INDEX (_NL_NUM_LC_ADDRESS));
  add_locale_string (&file, address->postal_fmt);
  add_locale_string (&file, address->country_name);
  add_locale_string (&file, address->country_post);
  add_locale_string (&file, address->country_ab2);
  add_locale_string (&file, address->country_ab3);
  add_locale_string (&file, address->country_car);
  add_locale_uint32 (&file, address->country_num);
  add_locale_string (&file, address->country_isbn);
  add_locale_string (&file, address->lang_name);
  add_locale_string (&file, address->lang_ab);
  add_locale_string (&file, address->lang_term);
  add_locale_string (&file, address->lang_lib);
  add_locale_string (&file, charmap->code_set_name);
  write_locale_data (output_path, LC_ADDRESS, "LC_ADDRESS", &file);
}


/* The parser for the LC_ADDRESS section of the locale definition.  */
void
address_read (struct linereader *ldfile, struct localedef_t *result,
	      const struct charmap_t *charmap, const char *repertoire_name,
	      int ignore_content)
{
  struct locale_address_t *address;
  struct token *now;
  struct token *arg;
  enum token_t nowtok;

  /* The rest of the line containing `LC_ADDRESS' must be free.  */
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
      handle_copy (ldfile, charmap, repertoire_name, result, tok_lc_address,
		   LC_ADDRESS, "LC_ADDRESS", ignore_content);
      return;
    }

  /* Prepare the data structures.  */
  address_startup (ldfile, result, ignore_content);
  address = result->categories[LC_ADDRESS].address;

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
	  if (address->cat != NULL)					      \
	    lr_error (ldfile, _("\
%s: field `%s' declared more than once"), "LC_ADDRESS", #cat);		      \
	  else if (!ignore_content && arg->val.str.startmb == NULL)	      \
	    {								      \
	      lr_error (ldfile, _("\
%s: unknown character in field `%s'"), "LC_ADDRESS", #cat);		      \
	      address->cat = "";					      \
	    }								      \
	  else if (!ignore_content)					      \
	    address->cat = arg->val.str.startmb;			      \
	  break

	  STR_ELEM (postal_fmt);
	  STR_ELEM (country_name);
	  STR_ELEM (country_post);
	  STR_ELEM (country_ab2);
	  STR_ELEM (country_ab3);
	  STR_ELEM (country_car);
	  STR_ELEM (lang_name);
	  STR_ELEM (lang_ab);
	  STR_ELEM (lang_term);
	  STR_ELEM (lang_lib);

#define INT_STR_ELEM(cat) \
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
	  if (arg->tok != tok_string && arg->tok != tok_number)		      \
	    goto err_label;						      \
	  if (address->cat != NULL)					      \
	    lr_error (ldfile, _("\
%s: field `%s' declared more than once"), "LC_ADDRESS", #cat);		      \
	  else if (!ignore_content && arg->tok == tok_string		      \
		   && arg->val.str.startmb == NULL)			      \
	    {								      \
	      lr_error (ldfile, _("\
%s: unknown character in field `%s'"), "LC_ADDRESS", #cat);		      \
	      address->cat = "";					      \
	    }								      \
	  else if (!ignore_content)					      \
	    {								      \
	      if (arg->tok == tok_string)				      \
		address->cat = arg->val.str.startmb;			      \
	      else							      \
		{							      \
		  char *numbuf = (char *) xmalloc (21);			      \
		  snprintf (numbuf, 21, "%ld", arg->val.num);		      \
		  address->cat = numbuf;				      \
		}							      \
	    }								      \
	  break

	  INT_STR_ELEM (country_isbn);

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
	  else if (address->cat != 0)					      \
	    lr_error (ldfile, _("\
%s: field `%s' declared more than once"), "LC_ADDRESS", #cat);		      \
	  else if (!ignore_content)					      \
	    address->cat = arg->val.num;				      \
	  break

	  INT_ELEM (country_num);

	case tok_end:
	  /* Next we assume `LC_ADDRESS'.  */
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok == tok_eof)
	    break;
	  if (arg->tok == tok_eol)
	    lr_error (ldfile, _("%s: incomplete `END' line"),
		      "LC_ADDRESS");
	  else if (arg->tok != tok_lc_address)
	    lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_ADDRESS");
	  lr_ignore_rest (ldfile, arg->tok == tok_lc_address);
	  return;

	default:
	err_label:
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_ADDRESS");
	}

      /* Prepare for the next round.  */
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }

  /* When we come here we reached the end of the file.  */
  lr_error (ldfile, _("%s: premature end of file"), "LC_ADDRESS");
}
