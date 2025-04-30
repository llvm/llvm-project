/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1995.

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
#include <sys/types.h>
#include <regex.h>
#include <string.h>
#include <stdint.h>
#include <sys/uio.h>

#include <assert.h>

#include "localedef.h"
#include "linereader.h"
#include "localeinfo.h"
#include "locfile.h"


/* The real definition of the struct for the LC_MESSAGES locale.  */
struct locale_messages_t
{
  const char *yesexpr;
  const char *noexpr;
  const char *yesstr;
  const char *nostr;
};


static void
messages_startup (struct linereader *lr, struct localedef_t *locale,
		  int ignore_content)
{
  if (!ignore_content)
    locale->categories[LC_MESSAGES].messages =
      (struct locale_messages_t *) xcalloc (1,
					    sizeof (struct locale_messages_t));

  if (lr != NULL)
    {
      lr->translate_strings = 1;
      lr->return_widestr = 0;
    }
}


void
messages_finish (struct localedef_t *locale, const struct charmap_t *charmap)
{
  struct locale_messages_t *messages
    = locale->categories[LC_MESSAGES].messages;
  int nothing = 0;

  /* Now resolve copying and also handle completely missing definitions.  */
  if (messages == NULL)
    {
      /* First see whether we were supposed to copy.  If yes, find the
	 actual definition.  */
      if (locale->copy_name[LC_MESSAGES] != NULL)
	{
	  /* Find the copying locale.  This has to happen transitively since
	     the locale we are copying from might also copying another one.  */
	  struct localedef_t *from = locale;

	  do
	    from = find_locale (LC_MESSAGES, from->copy_name[LC_MESSAGES],
				from->repertoire_name, charmap);
	  while (from->categories[LC_MESSAGES].messages == NULL
		 && from->copy_name[LC_MESSAGES] != NULL);

	  messages = locale->categories[LC_MESSAGES].messages
	    = from->categories[LC_MESSAGES].messages;
	}

      /* If there is still no definition issue an warning and create an
	 empty one.  */
      if (messages == NULL)
	{
	  record_warning (_("\
No definition for %s category found"), "LC_MESSAGES");
	  messages_startup (NULL, locale, 0);
	  messages = locale->categories[LC_MESSAGES].messages;
	  nothing = 1;
	}
    }

  /* The fields YESSTR and NOSTR are optional.  */
  if (messages->yesstr == NULL)
    messages->yesstr = "";
  if (messages->nostr == NULL)
    messages->nostr = "";

  if (messages->yesexpr == NULL)
    {
      if (! nothing)
	record_error (0, 0, _("%s: field `%s' undefined"),
		      "LC_MESSAGES", "yesexpr");
      messages->yesexpr = "^[yY]";
    }
  else if (messages->yesexpr[0] == '\0')
    {
      record_error (0, 0, _("\
%s: value for field `%s' must not be an empty string"),
		    "LC_MESSAGES", "yesexpr");
    }
  else
    {
      int result;
      regex_t re;

      /* Test whether it are correct regular expressions.  */
      result = regcomp (&re, messages->yesexpr, REG_EXTENDED);
      if (result != 0 && !be_quiet)
	{
	  char errbuf[BUFSIZ];

	  (void) regerror (result, &re, errbuf, BUFSIZ);
	  record_error (0, 0, _("\
%s: no correct regular expression for field `%s': %s"),
			"LC_MESSAGES", "yesexpr", errbuf);
	}
      else if (result != 0)
	regfree (&re);
    }

  if (messages->noexpr == NULL)
    {
      if (! nothing)
	record_error (0, 0, _("%s: field `%s' undefined"),
		      "LC_MESSAGES", "noexpr");
      messages->noexpr = "^[nN]";
    }
  else if (messages->noexpr[0] == '\0')
    {
      record_error (0, 0, _("\
%s: value for field `%s' must not be an empty string"),
		    "LC_MESSAGES", "noexpr");
    }
  else
    {
      int result;
      regex_t re;

      /* Test whether it are correct regular expressions.  */
      result = regcomp (&re, messages->noexpr, REG_EXTENDED);
      if (result != 0 && !be_quiet)
	{
	  char errbuf[BUFSIZ];

	  (void) regerror (result, &re, errbuf, BUFSIZ);
	  record_error (0, 0, _("\
%s: no correct regular expression for field `%s': %s"),
			"LC_MESSAGES", "noexpr", errbuf);
	}
      else if (result != 0)
	regfree (&re);
    }
}


void
messages_output (struct localedef_t *locale, const struct charmap_t *charmap,
		 const char *output_path)
{
  struct locale_messages_t *messages
    = locale->categories[LC_MESSAGES].messages;
  struct locale_file file;

  init_locale_data (&file, _NL_ITEM_INDEX (_NL_NUM_LC_MESSAGES));
  add_locale_string (&file, messages->yesexpr);
  add_locale_string (&file, messages->noexpr);
  add_locale_string (&file, messages->yesstr);
  add_locale_string (&file, messages->nostr);
  add_locale_string (&file, charmap->code_set_name);
  write_locale_data (output_path, LC_MESSAGES, "LC_MESSAGES", &file);
}


/* The parser for the LC_MESSAGES section of the locale definition.  */
void
messages_read (struct linereader *ldfile, struct localedef_t *result,
	       const struct charmap_t *charmap, const char *repertoire_name,
	       int ignore_content)
{
  struct repertoire_t *repertoire = NULL;
  struct locale_messages_t *messages;
  struct token *now;
  enum token_t nowtok;

  /* Get the repertoire we have to use.  */
  if (repertoire_name != NULL)
    repertoire = repertoire_read (repertoire_name);

  /* The rest of the line containing `LC_MESSAGES' must be free.  */
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
      handle_copy (ldfile, charmap, repertoire_name, result, tok_lc_messages,
		   LC_MESSAGES, "LC_MESSAGES", ignore_content);
      return;
    }

  /* Prepare the data structures.  */
  messages_startup (ldfile, result, ignore_content);
  messages = result->categories[LC_MESSAGES].messages;

  while (1)
    {
      struct token *arg;

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
	  if (messages->cat != NULL)					      \
	    {								      \
	      lr_error (ldfile, _("\
%s: field `%s' declared more than once"), "LC_MESSAGES", #cat);		      \
	      lr_ignore_rest (ldfile, 0);				      \
	      break;							      \
	    }								      \
	  now = lr_token (ldfile, charmap, result, repertoire, verbose);      \
	  if (now->tok != tok_string)					      \
	    goto syntax_error;						      \
	  else if (!ignore_content && now->val.str.startmb == NULL)	      \
	    {								      \
	      lr_error (ldfile, _("\
%s: unknown character in field `%s'"), "LC_MESSAGES", #cat);		      \
	      messages->cat = "";					      \
	    }								      \
	  else if (!ignore_content)					      \
	    messages->cat = now->val.str.startmb;			      \
	  break

	  STR_ELEM (yesexpr);
	  STR_ELEM (noexpr);
	  STR_ELEM (yesstr);
	  STR_ELEM (nostr);

	case tok_end:
	  /* Next we assume `LC_MESSAGES'.  */
	  arg = lr_token (ldfile, charmap, result, NULL, verbose);
	  if (arg->tok == tok_eof)
	    break;
	  if (arg->tok == tok_eol)
	    lr_error (ldfile, _("%s: incomplete `END' line"), "LC_MESSAGES");
	  else if (arg->tok != tok_lc_messages)
	    lr_error (ldfile, _("\
%1$s: definition does not end with `END %1$s'"), "LC_MESSAGES");
	  lr_ignore_rest (ldfile, arg->tok == tok_lc_messages);
	  return;

	default:
	syntax_error:
	  SYNTAX_ERROR (_("%s: syntax error"), "LC_MESSAGES");
	}

      /* Prepare for the next round.  */
      now = lr_token (ldfile, charmap, result, NULL, verbose);
      nowtok = now->tok;
    }

  /* When we come here we reached the end of the file.  */
  lr_error (ldfile, _("%s: premature end of file"), "LC_MESSAGES");
}
