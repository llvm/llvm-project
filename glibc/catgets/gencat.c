/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 1996.

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
# include "config.h"
#endif

#include <argp.h>
#include <assert.h>
#include <ctype.h>
#include <endian.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <iconv.h>
#include <langinfo.h>
#include <locale.h>
#include <libintl.h>
#include <limits.h>
#include <nl_types.h>
#include <obstack.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>

#include "version.h"

#include "catgetsinfo.h"


#define SWAPU32(w) \
  (((w) << 24) | (((w) & 0xff00) << 8) | (((w) >> 8) & 0xff00) | ((w) >> 24))

struct message_list
{
  int number;
  const char *message;

  const char *fname;
  size_t line;
  const char *symbol;

  struct message_list *next;
};


struct set_list
{
  int number;
  int deleted;
  struct message_list *messages;
  int last_message;

  const char *fname;
  size_t line;
  const char *symbol;

  struct set_list *next;
};


struct catalog
{
  struct set_list *all_sets;
  struct set_list *current_set;
  size_t total_messages;
  wint_t quote_char;
  int last_set;

  struct obstack mem_pool;
};


/* If non-zero force creation of new file, not using existing one.  */
static int force_new;

/* Name of output file.  */
static const char *output_name;

/* Name of generated C header file.  */
static const char *header_name;

/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

#define OPT_NEW 1

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { "header", 'H', N_("NAME"), 0,
    N_("Create C header file NAME containing symbol definitions") },
  { "new", OPT_NEW, NULL, 0,
    N_("Do not use existing catalog, force new output file") },
  { "output", 'o', N_("NAME"), 0, N_("Write output to file NAME") },
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("Generate message catalog.\
\vIf INPUT-FILE is -, input is read from standard input.  If OUTPUT-FILE\n\
is -, output is written to standard output.\n");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("\
-o OUTPUT-FILE [INPUT-FILE]...\n[OUTPUT-FILE [INPUT-FILE]...]");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help
};


/* Wrapper functions with error checking for standard functions.  */
#include <programs/xmalloc.h>

/* Prototypes for local functions.  */
static void error_print (void);
static struct catalog *read_input_file (struct catalog *current,
					const char *fname);
static void write_out (struct catalog *result, const char *output_name,
		       const char *header_name);
static struct set_list *find_set (struct catalog *current, int number);
static void normalize_line (const char *fname, size_t line, iconv_t cd,
			    wchar_t *string, wchar_t quote_char,
			    wchar_t escape_char);
static void read_old (struct catalog *catalog, const char *file_name);
static int open_conversion (const char *codesetp, iconv_t *cd_towcp,
			    iconv_t *cd_tombp, wchar_t *escape_charp);


int
main (int argc, char *argv[])
{
  struct catalog *result;
  int remaining;

  /* Set program name for messages.  */
  error_print_progname = error_print;

  /* Set locale via LC_ALL.  */
  setlocale (LC_ALL, "");

  /* Set the text message domain.  */
  textdomain (PACKAGE);

  /* Initialize local variables.  */
  result = NULL;

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  /* Determine output file.  */
  if (output_name == NULL)
    output_name = remaining < argc ? argv[remaining++] : "-";

  /* Process all input files.  */
  setlocale (LC_CTYPE, "C");
  if (remaining < argc)
    do
      result = read_input_file (result, argv[remaining]);
    while (++remaining < argc);
  else
    result = read_input_file (NULL, "-");

  /* Write out the result.  */
  if (result != NULL)
    write_out (result, output_name, header_name);

  return error_message_count != 0;
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'H':
      header_name = arg;
      break;
    case OPT_NEW:
      force_new = 1;
      break;
    case 'o':
      output_name = arg;
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


static char *
more_help (int key, const char *text, void *input)
{
  char *tp = NULL;
  switch (key)
    {
    case ARGP_KEY_HELP_EXTRA:
      /* We print some extra information.  */
      if (asprintf (&tp, gettext ("\
For bug reporting instructions, please see:\n\
%s.\n"), REPORT_BUGS_TO) < 0)
	return NULL;
      return tp;
    default:
      break;
    }
  return (char *) text;
}

/* Print the version information.  */
static void
print_version (FILE *stream, struct argp_state *state)
{
  fprintf (stream, "gencat %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


/* The address of this function will be assigned to the hook in the
   error functions.  */
static void
error_print (void)
{
  /* We don't want the program name to be printed in messages.  Emacs'
     compile.el does not like this.  */
}


static struct catalog *
read_input_file (struct catalog *current, const char *fname)
{
  FILE *fp;
  char *buf;
  size_t len;
  size_t line_number;
  wchar_t *wbuf;
  size_t wbufsize;
  iconv_t cd_towc = (iconv_t) -1;
  iconv_t cd_tomb = (iconv_t) -1;
  wchar_t escape_char = L'\\';
  char *codeset = NULL;

  if (strcmp (fname, "-") == 0 || strcmp (fname, "/dev/stdin") == 0)
    {
      fp = stdin;
      fname = gettext ("*standard input*");
    }
  else
    fp = fopen (fname, "r");
  if (fp == NULL)
    {
      error (0, errno, gettext ("cannot open input file `%s'"), fname);
      return current;
    }

  /* If we haven't seen anything yet, allocate result structure.  */
  if (current == NULL)
    {
      current = (struct catalog *) xcalloc (1, sizeof (*current));

#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
      obstack_init (&current->mem_pool);

      current->current_set = find_set (current, NL_SETD);
    }

  buf = NULL;
  len = 0;
  line_number = 0;

  wbufsize = 1024;
  wbuf = (wchar_t *) xmalloc (wbufsize);

  while (!feof (fp))
    {
      int continued;
      int used;
      size_t start_line = line_number + 1;
      char *this_line;

      do
	{
	  int act_len;

	  act_len = getline (&buf, &len, fp);
	  if (act_len <= 0)
	    break;
	  ++line_number;

	  /* It the line continued?  */
	  continued = 0;
	  if (buf[act_len - 1] == '\n')
	    {
	      --act_len;

	      /* There might be more than one backslash at the end of
		 the line.  Only if there is an odd number of them is
		 the line continued.  */
	      if (act_len > 0 && buf[act_len - 1] == '\\')
		{
		  int temp_act_len = act_len;

		  do
		    {
		      --temp_act_len;
		      continued = !continued;
		    }
		  while (temp_act_len > 0 && buf[temp_act_len - 1] == '\\');

		  if (continued)
		    --act_len;
		}
	    }

	  /* Append to currently selected line.  */
	  obstack_grow (&current->mem_pool, buf, act_len);
	}
      while (continued);

      obstack_1grow (&current->mem_pool, '\0');
      this_line = (char *) obstack_finish (&current->mem_pool);

      used = 0;
      if (this_line[0] == '$')
	{
	  if (isblank (this_line[1]))
	    {
	      int cnt = 1;
	      while (isblank (this_line[cnt]))
		++cnt;
	      if (strncmp (&this_line[cnt], "codeset=", 8) != 0)
		/* This is a comment line. Do nothing.  */;
	      else if (codeset != NULL)
		/* Ignore multiple codeset. */;
	      else
		{
		  int start = cnt + 8;
		  cnt = start;
		  while (this_line[cnt] != '\0' && !isspace (this_line[cnt]))
		    ++cnt;
		  if (cnt != start)
		    {
		      int len = cnt - start;
		      codeset = xmalloc (len + 1);
		      *((char *) mempcpy (codeset, &this_line[start], len))
			= '\0';
		    }
		}
	    }
	  else if (strncmp (&this_line[1], "set", 3) == 0)
	    {
	      int cnt = sizeof ("set");
	      int set_number;
	      const char *symbol = NULL;
	      while (isspace (this_line[cnt]))
		++cnt;

	      if (isdigit (this_line[cnt]))
		{
		  set_number = atol (&this_line[cnt]);

		  /* If the given number for the character set is
		     higher than any we used for symbolic set names
		     avoid clashing by using only higher numbers for
		     the following symbolic definitions.  */
		  if (set_number > current->last_set)
		    current->last_set = set_number;
		}
	      else
		{
		  /* See whether it is a reasonable identifier.  */
		  int start = cnt;
		  while (isalnum (this_line[cnt]) || this_line[cnt] == '_')
		    ++cnt;

		  if (cnt == start)
		    {
		      /* No correct character found.  */
		      error_at_line (0, 0, fname, start_line,
				     gettext ("illegal set number"));
		      set_number = 0;
		    }
		  else
		    {
		      /* We have found seomthing that looks like a
			 correct identifier.  */
		      struct set_list *runp;

		      this_line[cnt] = '\0';
		      used = 1;
		      symbol = &this_line[start];

		      /* Test whether the identifier was already used.  */
		      runp = current->all_sets;
		      while (runp != 0)
			if (runp->symbol != NULL
			    && strcmp (runp->symbol, symbol) == 0)
			  break;
			else
			  runp = runp->next;

		      if (runp != NULL)
			{
			  /* We cannot allow duplicate identifiers for
			     message sets.  */
			  error_at_line (0, 0, fname, start_line,
					 gettext ("duplicate set definition"));
			  error_at_line (0, 0, runp->fname, runp->line,
					 gettext ("\
this is the first definition"));
			  set_number = 0;
			}
		      else
			/* Allocate next free message set for identifier.  */
			set_number = ++current->last_set;
		    }
		}

	      if (set_number != 0)
		{
		  /* We found a legal set number.  */
		  current->current_set = find_set (current, set_number);
		  if (symbol != NULL)
		      used = 1;
		  current->current_set->symbol = symbol;
		  current->current_set->fname = fname;
		  current->current_set->line = start_line;
		}
	    }
	  else if (strncmp (&this_line[1], "delset", 6) == 0)
	    {
	      int cnt = sizeof ("delset");
	      while (isspace (this_line[cnt]))
		++cnt;

	      if (isdigit (this_line[cnt]))
		{
		  size_t set_number = atol (&this_line[cnt]);
		  struct set_list *set;

		  /* Mark the message set with the given number as
		     deleted.  */
		  set = find_set (current, set_number);
		  set->deleted = 1;
		}
	      else
		{
		  /* See whether it is a reasonable identifier.  */
		  int start = cnt;
		  while (isalnum (this_line[cnt]) || this_line[cnt] == '_')
		    ++cnt;

		  if (cnt == start)
		    error_at_line (0, 0, fname, start_line,
				   gettext ("illegal set number"));
		  else
		    {
		      const char *symbol;
		      struct set_list *runp;

		      this_line[cnt] = '\0';
		      used = 1;
		      symbol = &this_line[start];

		      /* We have a symbolic set name.  This name must
			 appear somewhere else in the catalogs read so
			 far.  */
		      for (runp = current->all_sets; runp != NULL;
			   runp = runp->next)
			{
			  if (strcmp (runp->symbol, symbol) == 0)
			    {
			      runp->deleted = 1;
			      break;
			    }
			}
		      if (runp == NULL)
			/* Name does not exist before.  */
			error_at_line (0, 0, fname, start_line,
				       gettext ("unknown set `%s'"), symbol);
		    }
		}
	    }
	  else if (strncmp (&this_line[1], "quote", 5) == 0)
	    {
	      char buf[2];
	      char *bufptr;
	      size_t buflen;
	      char *wbufptr;
	      size_t wbuflen;
	      int cnt;

	      cnt = sizeof ("quote");
	      while (isspace (this_line[cnt]))
		++cnt;

	      /* We need the conversion.  */
	      if (cd_towc == (iconv_t) -1
		  && open_conversion (codeset, &cd_towc, &cd_tomb,
				      &escape_char) != 0)
		/* Something is wrong.  */
		goto out;

	      /* Yes, the quote char can be '\0'; this means no quote
		 char.  The function using the information works on
		 wide characters so we have to convert it here.  */
	      buf[0] = this_line[cnt];
	      buf[1] = '\0';
	      bufptr = buf;
	      buflen = 2;

	      wbufptr = (char *) wbuf;
	      wbuflen = wbufsize;

	      /* Flush the state.  */
	      iconv (cd_towc, NULL, NULL, NULL, NULL);

	      iconv (cd_towc, &bufptr, &buflen, &wbufptr, &wbuflen);
	      if (buflen != 0 || (wchar_t *) wbufptr != &wbuf[2])
		error_at_line (0, 0, fname, start_line,
			       gettext ("invalid quote character"));
	      else
		/* Use the converted wide character.  */
		current->quote_char = wbuf[0];
	    }
	  else
	    {
	      int cnt;
	      cnt = 2;
	      while (this_line[cnt] != '\0' && !isspace (this_line[cnt]))
		++cnt;
	      this_line[cnt] = '\0';
	      error_at_line (0, 0, fname, start_line,
			     gettext ("unknown directive `%s': line ignored"),
			     &this_line[1]);
	    }
	}
      else if (isalnum (this_line[0]) || this_line[0] == '_')
	{
	  const char *ident = this_line;
	  char *line = this_line;
	  int message_number;

	  do
	    ++line;
	  while (line[0] != '\0' && !isspace (line[0]));
	  if (line[0] != '\0')
	    *line++ = '\0';	/* Terminate the identifier.  */

	  /* Now we found the beginning of the message itself.  */

	  if (isdigit (ident[0]))
	    {
	      struct message_list *runp;
	      struct message_list *lastp;

	      message_number = atoi (ident);

	      /* Find location to insert the new message.  */
	      runp = current->current_set->messages;
	      lastp = NULL;
	      while (runp != NULL)
		if (runp->number == message_number)
		  break;
		else
		  {
		    lastp = runp;
		    runp = runp->next;
		  }
	      if (runp != NULL)
		{
		  /* Oh, oh.  There is already a message with this
		     number in the message set.  */
		  if (runp->symbol == NULL)
		    {
		      /* The existing message had its number specified
			 by the user.  Fatal collision type uh, oh.  */
		      error_at_line (0, 0, fname, start_line,
				     gettext ("duplicated message number"));
		      error_at_line (0, 0, runp->fname, runp->line,
				     gettext ("this is the first definition"));
		      message_number = 0;
		    }
		  else
		    {
		      /* Collision was with number auto-assigned to a
			 symbolic.  Change existing symbolic number
			 and move to end the list (if not already there).  */
		      runp->number = ++current->current_set->last_message;

		      if (runp->next != NULL)
			{
			  struct message_list *endp;

			  if (lastp == NULL)
			    current->current_set->messages=runp->next;
			  else
			    lastp->next=runp->next;

			  endp = runp->next;
			  while (endp->next != NULL)
			    endp = endp->next;

			  endp->next = runp;
			  runp->next = NULL;
			}
		    }
		}
	      ident = NULL;	/* We don't have a symbol.  */

	      if (message_number != 0
		  && message_number > current->current_set->last_message)
		current->current_set->last_message = message_number;
	    }
	  else if (ident[0] != '\0')
	    {
	      struct message_list *runp;

	      /* Test whether the symbolic name was not used for
		 another message in this message set.  */
	      runp = current->current_set->messages;
	      while (runp != NULL)
		if (runp->symbol != NULL && strcmp (ident, runp->symbol) == 0)
		  break;
		else
		  runp = runp->next;
	      if (runp != NULL)
		{
		  /* The name is already used.  */
		  error_at_line (0, 0, fname, start_line, gettext ("\
duplicated message identifier"));
		  error_at_line (0, 0, runp->fname, runp->line,
				 gettext ("this is the first definition"));
		  message_number = 0;
		}
	      else
		/* Give the message the next unused number.  */
		message_number = ++current->current_set->last_message;
	    }
	  else
	    message_number = 0;

	  if (message_number != 0)
	    {
	      char *inbuf;
	      size_t inlen;
	      char *outbuf;
	      size_t outlen;
	      struct message_list *newp;
	      size_t line_len = strlen (line) + 1;
	      size_t ident_len = 0;

	      /* We need the conversion.  */
	      if (cd_towc == (iconv_t) -1
		  && open_conversion (codeset, &cd_towc, &cd_tomb,
				      &escape_char) != 0)
		/* Something is wrong.  */
		goto out;

	      /* Convert to a wide character string.  We have to
		 interpret escape sequences which will be impossible
		 without doing the conversion if the codeset of the
		 message is stateful.  */
	      while (1)
		{
		  inbuf = line;
		  inlen = line_len;
		  outbuf = (char *) wbuf;
		  outlen = wbufsize;

		  /* Flush the state.  */
		  iconv (cd_towc, NULL, NULL, NULL, NULL);

		  iconv (cd_towc, &inbuf, &inlen, &outbuf, &outlen);
		  if (inlen == 0)
		    {
		      /* The string is converted.  */
		      assert (outlen < wbufsize);
		      assert (wbuf[(wbufsize - outlen) / sizeof (wchar_t) - 1]
			      == L'\0');
		      break;
		    }

		  if (outlen != 0)
		    {
		      /* Something is wrong with this string, we ignore it.  */
		      error_at_line (0, 0, fname, start_line, gettext ("\
invalid character: message ignored"));
		      goto ignore;
		    }

		  /* The output buffer is too small.  */
		  wbufsize *= 2;
		  wbuf = (wchar_t *) xrealloc (wbuf, wbufsize);
		}

	      /* Strip quote characters, change escape sequences into
		 correct characters etc.  */
	      normalize_line (fname, start_line, cd_towc, wbuf,
			      current->quote_char, escape_char);

	      if (ident)
		ident_len = line - this_line;

	      /* Now the string is free of escape sequences.  Convert it
		 back into a multibyte character string.  First free the
		 memory allocated for the original string.  */
	      obstack_free (&current->mem_pool, this_line);

	      used = 1;	/* Yes, we use the line.  */

	      /* Now fill in the new string.  It should never happen that
		 the replaced string is longer than the original.  */
	      inbuf = (char *) wbuf;
	      inlen = (wcslen (wbuf) + 1) * sizeof (wchar_t);

	      outlen = obstack_room (&current->mem_pool);
	      obstack_blank (&current->mem_pool, outlen);
	      this_line = (char *) obstack_base (&current->mem_pool);
	      outbuf = this_line + ident_len;
	      outlen -= ident_len;

	      /* Flush the state.  */
	      iconv (cd_tomb, NULL, NULL, NULL, NULL);

	      iconv (cd_tomb, &inbuf, &inlen, &outbuf, &outlen);
	      if (inlen != 0)
		{
		  error_at_line (0, 0, fname, start_line,
				 gettext ("invalid line"));
		  goto ignore;
		}
	      assert (outbuf[-1] == '\0');

	      /* Free the memory in the obstack we don't use.  */
	      obstack_blank (&current->mem_pool, -(int) outlen);
	      line = obstack_finish (&current->mem_pool);

	      newp = (struct message_list *) xmalloc (sizeof (*newp));
	      newp->number = message_number;
	      newp->message = line + ident_len;
	      /* Remember symbolic name; is NULL if no is given.  */
	      newp->symbol = ident ? line : NULL;
	      /* Remember where we found the character.  */
	      newp->fname = fname;
	      newp->line = start_line;

	      /* Find place to insert to message.  We keep them in a
		 sorted single linked list.  */
	      if (current->current_set->messages == NULL
		  || current->current_set->messages->number > message_number)
		{
		  newp->next = current->current_set->messages;
		  current->current_set->messages = newp;
		}
	      else
		{
		  struct message_list *runp;
		  runp = current->current_set->messages;
		  while (runp->next != NULL)
		    if (runp->next->number > message_number)
		      break;
		    else
		      runp = runp->next;
		  newp->next = runp->next;
		  runp->next = newp;
		}
	    }
	  ++current->total_messages;
	}
      else
	{
	  size_t cnt;

	  cnt = 0;
	  /* See whether we have any non-white space character in this
	     line.  */
	  while (this_line[cnt] != '\0' && isspace (this_line[cnt]))
	    ++cnt;

	  if (this_line[cnt] != '\0')
	    /* Yes, some unknown characters found.  */
	    error_at_line (0, 0, fname, start_line,
			   gettext ("malformed line ignored"));
	}

    ignore:
      /* We can save the memory for the line if it was not used.  */
      if (!used)
	obstack_free (&current->mem_pool, this_line);
    }

  /* Close the conversion modules.  */
  iconv_close (cd_towc);
  iconv_close (cd_tomb);
  free (codeset);

 out:
  free (wbuf);

  if (fp != stdin)
    fclose (fp);
  return current;
}


static void
write_out (struct catalog *catalog, const char *output_name,
	   const char *header_name)
{
  /* Computing the "optimal" size.  */
  struct set_list *set_run;
  size_t best_total, best_size, best_depth;
  size_t act_size, act_depth;
  struct catalog_obj obj;
  struct obstack string_pool;
  const char *strings;
  size_t strings_size;
  uint32_t *array1, *array2;
  size_t cnt;
  int fd;

  /* If not otherwise told try to read file with existing
     translations.  */
  if (!force_new)
    read_old (catalog, output_name);

  /* Initialize best_size with a very high value.  */
  best_total = best_size = best_depth = UINT_MAX;

  /* We need some start size for testing.  Let's start with
     TOTAL_MESSAGES / 5, which theoretically provides a mean depth of
     5.  */
  act_size = 1 + catalog->total_messages / 5;

  /* We determine the size of a hash table here.  Because the message
     numbers can be chosen arbitrary by the programmer we cannot use
     the simple method of accessing the array using the message
     number.  The algorithm is based on the trivial hash function
     NUMBER % TABLE_SIZE, where collisions are stored in a second
     dimension up to TABLE_DEPTH.  We here compute TABLE_SIZE so that
     the needed space (= TABLE_SIZE * TABLE_DEPTH) is minimal.  */
  while (act_size <= best_total)
    {
      size_t deep[act_size];

      act_depth = 1;
      memset (deep, '\0', act_size * sizeof (size_t));
      set_run = catalog->all_sets;
      while (set_run != NULL)
	{
	  struct message_list *message_run;

	  message_run = set_run->messages;
	  while (message_run != NULL)
	    {
	      size_t idx = (message_run->number * set_run->number) % act_size;

	      ++deep[idx];
	      if (deep[idx] > act_depth)
		{
		  act_depth = deep[idx];
		  if (act_depth * act_size > best_total)
		    break;
		}
	      message_run = message_run->next;
	    }
	  set_run = set_run->next;
	}

      if (act_depth * act_size <= best_total)
	{
	  /* We have found a better solution.  */
	  best_total = act_depth * act_size;
	  best_size = act_size;
	  best_depth = act_depth;
	}

      ++act_size;
    }

  /* let's be prepared for an empty message file.  */
  if (best_size == UINT_MAX)
    {
      best_size = 1;
      best_depth = 1;
    }

  /* OK, now we have the size we will use.  Fill in the header, build
     the table and the second one with swapped byte order.  */
  obj.magic = CATGETS_MAGIC;
  obj.plane_size = best_size;
  obj.plane_depth = best_depth;

  /* Allocate room for all needed arrays.  */
  array1 =
    (uint32_t *) alloca (best_size * best_depth * sizeof (uint32_t) * 3);
  memset (array1, '\0', best_size * best_depth * sizeof (uint32_t) * 3);
  array2
    = (uint32_t *) alloca (best_size * best_depth * sizeof (uint32_t) * 3);
  obstack_init (&string_pool);

  set_run = catalog->all_sets;
  while (set_run != NULL)
    {
      struct message_list *message_run;

      message_run = set_run->messages;
      while (message_run != NULL)
	{
	  size_t idx = (((message_run->number * set_run->number) % best_size)
			* 3);
	  /* Determine collision depth.  */
	  while (array1[idx] != 0)
	    idx += best_size * 3;

	  /* Store set number, message number and pointer into string
	     space, relative to the first string.  */
	  array1[idx + 0] = set_run->number;
	  array1[idx + 1] = message_run->number;
	  array1[idx + 2] = obstack_object_size (&string_pool);

	  /* Add current string to the continuous space containing all
	     strings.  */
	  obstack_grow0 (&string_pool, message_run->message,
			 strlen (message_run->message));

	  message_run = message_run->next;
	}

      set_run = set_run->next;
    }
  strings_size = obstack_object_size (&string_pool);
  strings = obstack_finish (&string_pool);

  /* Compute ARRAY2 by changing the byte order.  */
  for (cnt = 0; cnt < best_size * best_depth * 3; ++cnt)
    array2[cnt] = SWAPU32 (array1[cnt]);

  /* Now we can write out the whole data.  */
  if (strcmp (output_name, "-") == 0
      || strcmp (output_name, "/dev/stdout") == 0)
    fd = STDOUT_FILENO;
  else
    {
      fd = creat (output_name, 0666);
      if (fd < 0)
	error (EXIT_FAILURE, errno, gettext ("cannot open output file `%s'"),
	       output_name);
    }

  /* Write out header.  */
  write (fd, &obj, sizeof (obj));

  /* We always write out the little endian version of the index
     arrays.  */
#if __BYTE_ORDER == __LITTLE_ENDIAN
  write (fd, array1, best_size * best_depth * sizeof (uint32_t) * 3);
  write (fd, array2, best_size * best_depth * sizeof (uint32_t) * 3);
#elif __BYTE_ORDER == __BIG_ENDIAN
  write (fd, array2, best_size * best_depth * sizeof (uint32_t) * 3);
  write (fd, array1, best_size * best_depth * sizeof (uint32_t) * 3);
#else
# error Cannot handle __BYTE_ORDER byte order
#endif

  /* Finally write the strings.  */
  write (fd, strings, strings_size);

  if (fd != STDOUT_FILENO)
    close (fd);

  /* If requested now write out the header file.  */
  if (header_name != NULL)
    {
      int first = 1;
      FILE *fp;

      /* Open output file.  "-" or "/dev/stdout" means write to
	 standard output.  */
      if (strcmp (header_name, "-") == 0
	  || strcmp (header_name, "/dev/stdout") == 0)
	fp = stdout;
      else
	{
	  fp = fopen (header_name, "w");
	  if (fp == NULL)
	    error (EXIT_FAILURE, errno,
		   gettext ("cannot open output file `%s'"), header_name);
	}

      /* Iterate over all sets and all messages.  */
      set_run = catalog->all_sets;
      while (set_run != NULL)
	{
	  struct message_list *message_run;

	  /* If the current message set has a symbolic name write this
	     out first.  */
	  if (set_run->symbol != NULL)
	    fprintf (fp, "%s#define %sSet %#x\t/* %s:%zu */\n",
		     first ? "" : "\n", set_run->symbol, set_run->number - 1,
		     set_run->fname, set_run->line);
	  first = 0;

	  message_run = set_run->messages;
	  while (message_run != NULL)
	    {
	      /* If the current message has a symbolic name write
		 #define out.  But we have to take care for the set
		 not having a symbolic name.  */
	      if (message_run->symbol != NULL)
		{
		  if (set_run->symbol == NULL)
		    fprintf (fp, "#define AutomaticSet%d%s %#x\t/* %s:%zu */\n",
			     set_run->number, message_run->symbol,
			     message_run->number, message_run->fname,
			     message_run->line);
		  else
		    fprintf (fp, "#define %s%s %#x\t/* %s:%zu */\n",
			     set_run->symbol, message_run->symbol,
			     message_run->number, message_run->fname,
			     message_run->line);
		}

	      message_run = message_run->next;
	    }

	  set_run = set_run->next;
	}

      if (fp != stdout)
	fclose (fp);
    }
}


static struct set_list *
find_set (struct catalog *current, int number)
{
  struct set_list *result = current->all_sets;

  /* We must avoid set number 0 because a set of this number signals
     in the tables that the entry is not occupied.  */
  ++number;

  while (result != NULL)
    if (result->number == number)
      return result;
    else
      result = result->next;

  /* Prepare new message set.  */
  result = (struct set_list *) xcalloc (1, sizeof (*result));
  result->number = number;
  result->next = current->all_sets;
  current->all_sets = result;

  return result;
}


/* Normalize given string *in*place* by processing escape sequences
   and quote characters.  */
static void
normalize_line (const char *fname, size_t line, iconv_t cd, wchar_t *string,
		wchar_t quote_char, wchar_t escape_char)
{
  int is_quoted;
  wchar_t *rp = string;
  wchar_t *wp = string;

  if (quote_char != L'\0' && *rp == quote_char)
    {
      is_quoted = 1;
      ++rp;
    }
  else
    is_quoted = 0;

  while (*rp != L'\0')
    if (*rp == quote_char)
      /* We simply end the string when we find the first time an
	 not-escaped quote character.  */
	break;
    else if (*rp == escape_char)
      {
	++rp;
	if (quote_char != L'\0' && *rp == quote_char)
	  /* This is an extension to XPG.  */
	  *wp++ = *rp++;
	else
	  /* Recognize escape sequences.  */
	  switch (*rp)
	    {
	    case L'n':
	      *wp++ = L'\n';
	      ++rp;
	      break;
	    case L't':
	      *wp++ = L'\t';
	      ++rp;
	      break;
	    case L'v':
	      *wp++ = L'\v';
	      ++rp;
	      break;
	    case L'b':
	      *wp++ = L'\b';
	      ++rp;
	      break;
	    case L'r':
	      *wp++ = L'\r';
	      ++rp;
	      break;
	    case L'f':
	      *wp++ = L'\f';
	      ++rp;
	      break;
	    case L'0' ... L'7':
	      {
		int number;
		char cbuf[2];
		char *cbufptr;
		size_t cbufin;
		wchar_t wcbuf[2];
		char *wcbufptr;
		size_t wcbufin;

		number = *rp++ - L'0';
		while (number <= (255 / 8) && *rp >= L'0' && *rp <= L'7')
		  {
		    number *= 8;
		    number += *rp++ - L'0';
		  }

		cbuf[0] = (char) number;
		cbuf[1] = '\0';
		cbufptr = cbuf;
		cbufin = 2;

		wcbufptr = (char *) wcbuf;
		wcbufin = sizeof (wcbuf);

		/* Flush the state.  */
		iconv (cd, NULL, NULL, NULL, NULL);

		iconv (cd, &cbufptr, &cbufin, &wcbufptr, &wcbufin);
		if (cbufptr != &cbuf[2] || (wchar_t *) wcbufptr != &wcbuf[2])
		  error_at_line (0, 0, fname, line,
				 gettext ("invalid escape sequence"));
		else
		  *wp++ = wcbuf[0];
	      }
	      break;
	    default:
	      if (*rp == escape_char)
		{
		  *wp++ = escape_char;
		  ++rp;
		}
	      else
		{
		  /* Simply ignore the backslash character.  */
		}
	      break;
	    }
      }
    else
      *wp++ = *rp++;

  /* If we saw a quote character at the beginning we expect another
     one at the end.  */
  if (is_quoted && *rp != quote_char)
    error_at_line (0, 0, fname, line, gettext ("unterminated message"));

  /* Terminate string.  */
  *wp = L'\0';
  return;
}


static void
read_old (struct catalog *catalog, const char *file_name)
{
  struct catalog_info old_cat_obj;
  struct set_list *set = NULL;
  int last_set = -1;
  size_t cnt;

  /* Try to open catalog, but don't look through the NLSPATH.  */
  if (__open_catalog (file_name, NULL, NULL, &old_cat_obj) != 0)
    {
      if (errno == ENOENT)
	/* No problem, the catalog simply does not exist.  */
	return;
      else
	error (EXIT_FAILURE, errno,
	       gettext ("while opening old catalog file"));
    }

  /* OK, we have the catalog loaded.  Now read all messages and merge
     them.  When set and message number clash for any message the new
     one is used.  If the new one is empty it indicates that the
     message should be deleted.  */
  for (cnt = 0; cnt < old_cat_obj.plane_size * old_cat_obj.plane_depth; ++cnt)
    {
      struct message_list *message, *last;

      if (old_cat_obj.name_ptr[cnt * 3 + 0] == 0)
	/* No message in this slot.  */
	continue;

      if (old_cat_obj.name_ptr[cnt * 3 + 0] - 1 != (uint32_t) last_set)
	{
	  last_set = old_cat_obj.name_ptr[cnt * 3 + 0] - 1;
	  set = find_set (catalog, old_cat_obj.name_ptr[cnt * 3 + 0] - 1);
	}

      last = NULL;
      message = set->messages;
      while (message != NULL)
	{
	  if ((uint32_t) message->number >= old_cat_obj.name_ptr[cnt * 3 + 1])
	    break;
	  last = message;
	  message = message->next;
	}

      if (message == NULL
	  || (uint32_t) message->number > old_cat_obj.name_ptr[cnt * 3 + 1])
	{
	  /* We have found a message which is not yet in the catalog.
	     Insert it at the right position.  */
	  struct message_list *newp;

	  newp = (struct message_list *) xmalloc (sizeof (*newp));
	  newp->number = old_cat_obj.name_ptr[cnt * 3 + 1];
	  newp->message =
	    &old_cat_obj.strings[old_cat_obj.name_ptr[cnt * 3 + 2]];
	  newp->fname = NULL;
	  newp->line = 0;
	  newp->symbol = NULL;
	  newp->next = message;

	  if (last == NULL)
	    set->messages = newp;
	  else
	    last->next = newp;

	  ++catalog->total_messages;
	}
      else if (*message->message == '\0')
	{
	  /* The new empty message has overridden the old one thus
	     "deleting" it as required.  Now remove the empty remains. */
	  if (last == NULL)
	    set->messages = message->next;
	  else
	    last->next = message->next;
	}
    }
}


static int
open_conversion (const char *codeset, iconv_t *cd_towcp, iconv_t *cd_tombp,
		 wchar_t *escape_charp)
{
  char buf[2];
  char *bufptr;
  size_t bufsize;
  wchar_t wbuf[2];
  char *wbufptr;
  size_t wbufsize;

  /* If the input file does not specify the codeset use the locale's.  */
  if (codeset == NULL)
    {
      setlocale (LC_ALL, "");
      codeset = nl_langinfo (CODESET);
      setlocale (LC_ALL, "C");
    }

  /* Get the conversion modules.  */
  *cd_towcp = iconv_open ("WCHAR_T", codeset);
  *cd_tombp = iconv_open (codeset, "WCHAR_T");
  if (*cd_towcp == (iconv_t) -1 || *cd_tombp == (iconv_t) -1)
    {
      error (0, 0, gettext ("conversion modules not available"));
      if (*cd_towcp != (iconv_t) -1)
	iconv_close (*cd_towcp);

      return 1;
    }

  /* One special case for historical reasons is the backslash
     character.  In some codesets the byte value 0x5c is not mapped to
     U005c in Unicode.  These charsets then don't have a backslash
     character at all.  Therefore we have to live with whatever the
     codeset provides and recognize, instead of the U005c, the character
     the byte value 0x5c is mapped to.  */
  buf[0] = '\\';
  buf[1] = '\0';
  bufptr = buf;
  bufsize = 2;

  wbufptr = (char *) wbuf;
  wbufsize = sizeof (wbuf);

  iconv (*cd_towcp, &bufptr, &bufsize, &wbufptr, &wbufsize);
  if (bufsize != 0 || wbufsize != 0)
    {
      /* Something went wrong, we couldn't convert the byte 0x5c.  Go
	 on with using U005c.  */
      error (0, 0, gettext ("cannot determine escape character"));
      *escape_charp = L'\\';
    }
  else
    *escape_charp = wbuf[0];

  return 0;
}
