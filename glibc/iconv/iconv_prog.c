/* Convert text in given files from the specified from-set to the to-set.
   Copyright (C) 1998-2021 Free Software Foundation, Inc.
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

#include <argp.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <iconv.h>
#include <langinfo.h>
#include <locale.h>
#include <search.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libintl.h>
#ifdef _POSIX_MAPPED_FILES
# include <sys/mman.h>
#endif
#include <charmap.h>
#include <gconv_int.h>
#include "iconv_prog.h"
#include "iconvconfig.h"
#include "gconv_charset.h"

/* Get libc version number.  */
#include "../version.h"

#define PACKAGE _libc_intl_domainname


/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

#define OPT_VERBOSE	1000
#define OPT_LIST	'l'

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { NULL, 0, NULL, 0, N_("Input/Output format specification:") },
  { "from-code", 'f', N_("NAME"), 0, N_("encoding of original text") },
  { "to-code", 't', N_("NAME"), 0, N_("encoding for output") },
  { NULL, 0, NULL, 0, N_("Information:") },
  { "list", 'l', NULL, 0, N_("list all known coded character sets") },
  { NULL, 0, NULL, 0, N_("Output control:") },
  { NULL, 'c', NULL, 0, N_("omit invalid characters from output") },
  { "output", 'o', N_("FILE"), 0, N_("output file") },
  { "silent", 's', NULL, 0, N_("suppress warnings") },
  { "verbose", OPT_VERBOSE, NULL, 0, N_("print progress information") },
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("\
Convert encoding of given files from one encoding to another.");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("[FILE...]");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help
};

/* Code sets to convert from and to respectively.  An empty string as the
   default causes the 'iconv_open' function to look up the charset of the
   currently selected locale and use it.  */
static const char *from_code = "";
static const char *to_code = "";

/* File to write output to.  If NULL write to stdout.  */
static const char *output_file;

/* Nonzero if list of all coded character sets is wanted.  */
static int list;

/* If nonzero omit invalid character from output.  */
int omit_invalid;

/* Prototypes for the functions doing the actual work.  */
static int process_block (iconv_t cd, char *addr, size_t len, FILE **output,
			  const char *output_file);
static int process_fd (iconv_t cd, int fd, FILE **output,
		       const char *output_file);
static int process_file (iconv_t cd, FILE *input, FILE **output,
			 const char *output_file);
static void print_known_names (void);


int
main (int argc, char *argv[])
{
  int status = EXIT_SUCCESS;
  int remaining;
  __gconv_t cd;
  struct charmap_t *from_charmap = NULL;
  struct charmap_t *to_charmap = NULL;

  /* Set locale via LC_ALL.  */
  setlocale (LC_ALL, "");

  /* Set the text message domain.  */
  textdomain (_libc_intl_domainname);

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  /* List all coded character sets if wanted.  */
  if (list)
    {
      print_known_names ();
      exit (EXIT_SUCCESS);
    }

  /* POSIX 1003.2b introduces a silly thing: the arguments to -t anf -f
     can be file names of charmaps.  In this case iconv will have to read
     those charmaps and use them to do the conversion.  But there are
     holes in the specification.  There is nothing said that if -f is a
     charmap filename that -t must be, too.  And vice versa.  There is
     also no word about the symbolic names used.  What if they don't
     match?  */
  if (strchr (from_code, '/') != NULL)
    /* The from-name might be a charmap file name.  Try reading the
       file.  */
    from_charmap = charmap_read (from_code, /*0, 1*/1, 0, 0, 0);

  if (strchr (to_code, '/') != NULL)
    /* The to-name might be a charmap file name.  Try reading the
       file.  */
    to_charmap = charmap_read (to_code, /*0, 1,*/1, 0, 0, 0);


  /* At this point we have to handle two cases.  The first one is
     where a charmap is used for the from- or to-charset, or both.  We
     handle this special since it is very different from the sane way of
     doing things.  The other case allows converting using the iconv()
     function.  */
  if (from_charmap != NULL || to_charmap != NULL)
    /* Construct the conversion table and do the conversion.  */
    status = charmap_conversion (from_code, from_charmap, to_code, to_charmap,
				 argc, remaining, argv, output_file);
  else
    {
      struct gconv_spec conv_spec;
      int res;

      if (__gconv_create_spec (&conv_spec, from_code, to_code) == NULL)
        {
          error (EXIT_FAILURE, errno,
                 _("failed to start conversion processing"));
          exit (1);
        }

      if (omit_invalid)
        conv_spec.ignore = true;

      /* Let's see whether we have these coded character sets.  */
      res = __gconv_open (&conv_spec, &cd, 0);

      __gconv_destroy_spec (&conv_spec);

      if (res != __GCONV_OK)
	{
	  if (errno == EINVAL)
	    {
	      /* Try to be nice with the user and tell her which of the
		 two encoding names is wrong.  This is possible because
		 all supported encodings can be converted from/to Unicode,
		 in other words, because the graph of encodings is
		 connected.  */
	      bool from_wrong =
		(iconv_open ("UTF-8", from_code) == (iconv_t) -1
		 && errno == EINVAL);
	      bool to_wrong =
		(iconv_open (to_code, "UTF-8") == (iconv_t) -1
		 && errno == EINVAL);
	      const char *from_pretty =
		(from_code[0] ? from_code : nl_langinfo (CODESET));
	      const char *to_pretty =
		(to_code[0] ? to_code : nl_langinfo (CODESET));

	      if (from_wrong)
		{
		  if (to_wrong)
		    error (0, 0,
			   _("\
conversions from `%s' and to `%s' are not supported"),
			   from_pretty, to_pretty);
		  else
		    error (0, 0,
			   _("conversion from `%s' is not supported"),
			   from_pretty);
		}
	      else
		{
		  if (to_wrong)
		    error (0, 0,
			   _("conversion to `%s' is not supported"),
			   to_pretty);
		  else
		    error (0, 0,
			   _("conversion from `%s' to `%s' is not supported"),
			   from_pretty, to_pretty);
		}

	      argp_help (&argp, stderr, ARGP_HELP_SEE,
			 program_invocation_short_name);
	      exit (1);
	    }
	  else
	    error (EXIT_FAILURE, errno,
		   _("failed to start conversion processing"));
	}

      /* The output file.  Will be opened when we are ready to produce
	 output.  */
      FILE *output = NULL;

      /* Now process the remaining files.  Write them to stdout or the file
	 specified with the `-o' parameter.  If we have no file given as
	 the parameter process all from stdin.  */
      if (remaining == argc)
	{
	  if (process_file (cd, stdin, &output, output_file) != 0)
	    status = EXIT_FAILURE;
	}
      else
	do
	  {
#ifdef _POSIX_MAPPED_FILES
	    struct stat64 st;
	    char *addr;
#endif
	    int fd, ret;

	    if (verbose)
	      fprintf (stderr, "%s:\n", argv[remaining]);
	    if (strcmp (argv[remaining], "-") == 0)
	      fd = 0;
	    else
	      {
		fd = open (argv[remaining], O_RDONLY);

		if (fd == -1)
		  {
		    error (0, errno, _("cannot open input file `%s'"),
			   argv[remaining]);
		    status = EXIT_FAILURE;
		    continue;
		  }
	      }

#ifdef _POSIX_MAPPED_FILES
	    /* We have possibilities for reading the input file.  First try
	       to mmap() it since this will provide the fastest solution.  */
	    if (fstat64 (fd, &st) == 0
		&& ((addr = mmap (NULL, st.st_size, PROT_READ, MAP_PRIVATE,
				  fd, 0)) != MAP_FAILED))
	      {
		/* Yes, we can use mmap().  The descriptor is not needed
		   anymore.  */
		if (close (fd) != 0)
		  error (EXIT_FAILURE, errno,
			 _("error while closing input `%s'"),
			 argv[remaining]);

		ret = process_block (cd, addr, st.st_size, &output,
				     output_file);

		/* We don't need the input data anymore.  */
		munmap ((void *) addr, st.st_size);

		if (ret != 0)
		  {
		    status = EXIT_FAILURE;

		    if (ret < 0)
		      /* We cannot go on with producing output since it might
			 lead to problem because the last output might leave
			 the output stream in an undefined state.  */
		      break;
		  }
	      }
	    else
#endif	/* _POSIX_MAPPED_FILES */
	      {
		/* Read the file in pieces.  */
		ret = process_fd (cd, fd, &output, output_file);

		/* Now close the file.  */
		close (fd);

		if (ret != 0)
		  {
		    /* Something went wrong.  */
		    status = EXIT_FAILURE;

		    if (ret < 0)
		      /* We cannot go on with producing output since it might
			 lead to problem because the last output might leave
			 the output stream in an undefined state.  */
		      break;
		  }
	      }
	  }
	while (++remaining < argc);

      /* Close the output file now.  */
      if (output != NULL && fclose (output))
	error (EXIT_FAILURE, errno, _("error while closing output file"));
    }

  return status;
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'f':
      from_code = arg;
      break;
    case 't':
      to_code = arg;
      break;
    case 'o':
      output_file = arg;
      break;
    case 's':
      /* Nothing, for now at least.  We are not giving out any information
	 about missing character or so.  */
      break;
    case 'c':
      /* Omit invalid characters from output.  */
      omit_invalid = 1;
      break;
    case OPT_VERBOSE:
      verbose = 1;
      break;
    case OPT_LIST:
      list = 1;
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
  fprintf (stream, "iconv %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


static int
write_output (const char *outbuf, const char *outptr, FILE **output,
	      const char *output_file)
{
  /* We have something to write out.  */
  int errno_save = errno;

  if (*output == NULL)
    {
      /* Determine output file.  */
      if (output_file != NULL && strcmp (output_file, "-") != 0)
	{
	  *output = fopen (output_file, "w");
	  if (*output == NULL)
	    error (EXIT_FAILURE, errno, _("cannot open output file"));
	}
      else
	*output = stdout;
    }

  if (fwrite (outbuf, 1, outptr - outbuf, *output) < (size_t) (outptr - outbuf)
      || ferror (*output))
    {
      /* Error occurred while printing the result.  */
      error (0, 0, _("\
conversion stopped due to problem in writing the output"));
      return -1;
    }

  errno = errno_save;

  return 0;
}


static int
process_block (iconv_t cd, char *addr, size_t len, FILE **output,
	       const char *output_file)
{
#define OUTBUF_SIZE	32768
  const char *start = addr;
  char outbuf[OUTBUF_SIZE];
  char *outptr;
  size_t outlen;
  size_t n;
  int ret = 0;

  while (len > 0)
    {
      outptr = outbuf;
      outlen = OUTBUF_SIZE;
      n = iconv (cd, &addr, &len, &outptr, &outlen);

      if (n == (size_t) -1 && omit_invalid && errno == EILSEQ)
	{
	  ret = 1;
	  if (len == 0)
	    n = 0;
	  else
	    errno = E2BIG;
	}

      if (outptr != outbuf)
	{
	  ret = write_output (outbuf, outptr, output, output_file);
	  if (ret != 0)
	    break;
	}

      if (n != (size_t) -1)
	{
	  /* All the input test is processed.  For state-dependent
	     character sets we have to flush the state now.  */
	  outptr = outbuf;
	  outlen = OUTBUF_SIZE;
	  n = iconv (cd, NULL, NULL, &outptr, &outlen);

	  if (outptr != outbuf)
	    {
	      ret = write_output (outbuf, outptr, output, output_file);
	      if (ret != 0)
		break;
	    }

	  if (n != (size_t) -1)
	    break;

	  if (omit_invalid && errno == EILSEQ)
	    {
	      ret = 1;
	      break;
	    }
	}

      if (errno != E2BIG)
	{
	  /* iconv() ran into a problem.  */
	  switch (errno)
	    {
	    case EILSEQ:
	      if (! omit_invalid)
		error (0, 0, _("illegal input sequence at position %ld"),
		       (long int) (addr - start));
	      break;
	    case EINVAL:
	      error (0, 0, _("\
incomplete character or shift sequence at end of buffer"));
	      break;
	    case EBADF:
	      error (0, 0, _("internal error (illegal descriptor)"));
	      break;
	    default:
	      error (0, 0, _("unknown iconv() error %d"), errno);
	      break;
	    }

	  return -1;
	}
    }

  return ret;
}


static int
process_fd (iconv_t cd, int fd, FILE **output, const char *output_file)
{
  /* we have a problem with reading from a desriptor since we must not
     provide the iconv() function an incomplete character or shift
     sequence at the end of the buffer.  Since we have to deal with
     arbitrary encodings we must read the whole text in a buffer and
     process it in one step.  */
  static char *inbuf = NULL;
  static size_t maxlen = 0;
  char *inptr = NULL;
  size_t actlen = 0;

  while (actlen < maxlen)
    {
      ssize_t n = read (fd, inptr, maxlen - actlen);

      if (n == 0)
	/* No more text to read.  */
	break;

      if (n == -1)
	{
	  /* Error while reading.  */
	  error (0, errno, _("error while reading the input"));
	  return -1;
	}

      inptr += n;
      actlen += n;
    }

  if (actlen == maxlen)
    while (1)
      {
	ssize_t n;
	char *new_inbuf;

	/* Increase the buffer.  */
	new_inbuf = (char *) realloc (inbuf, maxlen + 32768);
	if (new_inbuf == NULL)
	  {
	    error (0, errno, _("unable to allocate buffer for input"));
	    return -1;
	  }
	inbuf = new_inbuf;
	maxlen += 32768;
	inptr = inbuf + actlen;

	do
	  {
	    n = read (fd, inptr, maxlen - actlen);

	    if (n == 0)
	      /* No more text to read.  */
	      break;

	    if (n == -1)
	      {
		/* Error while reading.  */
		error (0, errno, _("error while reading the input"));
		return -1;
	      }

	    inptr += n;
	    actlen += n;
	  }
	while (actlen < maxlen);

	if (n == 0)
	  /* Break again so we leave both loops.  */
	  break;
      }

  /* Now we have all the input in the buffer.  Process it in one run.  */
  return process_block (cd, inbuf, actlen, output, output_file);
}


static int
process_file (iconv_t cd, FILE *input, FILE **output, const char *output_file)
{
  /* This should be safe since we use this function only for `stdin' and
     we haven't read anything so far.  */
  return process_fd (cd, fileno (input), output, output_file);
}


/* Print all known character sets/encodings.  */
static void *printlist;
static size_t column;
static int not_first;

static void
insert_print_list (const void *nodep, VISIT value, int level)
{
  if (value == leaf || value == postorder)
    {
      const struct gconv_alias *s = *(const struct gconv_alias **) nodep;
      tsearch (s->fromname, &printlist, (__compar_fn_t) strverscmp);
    }
}

static void
do_print_human  (const void *nodep, VISIT value, int level)
{
  if (value == leaf || value == postorder)
    {
      const char *s = *(const char **) nodep;
      size_t len = strlen (s);
      size_t cnt;

      while (len > 0 && s[len - 1] == '/')
	--len;

      for (cnt = 0; cnt < len; ++cnt)
	if (isalnum (s[cnt]))
	  break;
      if (cnt == len)
	return;

      if (not_first)
	{
	  putchar (',');
	  ++column;

	  if (column > 2 && column + len > 77)
	    {
	      fputs ("\n  ", stdout);
	      column = 2;
	    }
	  else
	    {
	      putchar (' ');
	      ++column;
	    }
	}
      else
	not_first = 1;

      fwrite (s, len, 1, stdout);
      column += len;
    }
}

static void
do_print  (const void *nodep, VISIT value, int level)
{
  if (value == leaf || value == postorder)
    {
      const char *s = *(const char **) nodep;

      puts (s);
    }
}

static void
add_known_names (struct gconv_module *node)
{
  if (node->left != NULL)
    add_known_names (node->left);
  if (node->right != NULL)
    add_known_names (node->right);
  do
    {
      if (strcmp (node->from_string, "INTERNAL") != 0)
	tsearch (node->from_string, &printlist, (__compar_fn_t) strverscmp);
      if (strcmp (node->to_string, "INTERNAL") != 0)
	tsearch (node->to_string, &printlist, (__compar_fn_t) strverscmp);

      node = node->same;
    }
  while (node != NULL);
}


static void
insert_cache (void)
{
  const struct gconvcache_header *header;
  const char *strtab;
  const struct hash_entry *hashtab;
  size_t cnt;

  header = (const struct gconvcache_header *) __gconv_get_cache ();
  strtab = (char *) header + header->string_offset;
  hashtab = (struct hash_entry *) ((char *) header + header->hash_offset);

  for (cnt = 0; cnt < header->hash_size; ++cnt)
    if (hashtab[cnt].string_offset != 0)
      {
	const char *str = strtab + hashtab[cnt].string_offset;

	if (strcmp (str, "INTERNAL") != 0)
	  tsearch (str, &printlist, (__compar_fn_t) strverscmp);
      }
}


static void
print_known_names (void)
{
  iconv_t h;
  void *cache;

  /* We must initialize the internal databases first.  */
  h = iconv_open ("L1", "L1");
  iconv_close (h);

  /* See whether we have a cache.  */
  cache = __gconv_get_cache ();
  if (cache != NULL)
    /* Yep, use only this information.  */
    insert_cache ();
  else
    {
      struct gconv_module *modules;

      /* No, then use the information read from the gconv-modules file.
	 First add the aliases.  */
      twalk (__gconv_get_alias_db (), insert_print_list);

      /* Add the from- and to-names from the known modules.  */
      modules = __gconv_get_modules_db ();
      if (modules != NULL)
	add_known_names (modules);
    }

  bool human_readable = isatty (fileno (stdout));

  if (human_readable)
    fputs (_("\
The following list contains all the coded character sets known.  This does\n\
not necessarily mean that all combinations of these names can be used for\n\
the FROM and TO command line parameters.  One coded character set can be\n\
listed with several different names (aliases).\n\n  "), stdout);

  /* Now print the collected names.  */
  column = 2;
  twalk (printlist, human_readable ? do_print_human : do_print);

  if (human_readable && column != 0)
    puts ("");
}
