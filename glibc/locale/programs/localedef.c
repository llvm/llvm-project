/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1995.

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

#include <argp.h>
#include <errno.h>
#include <fcntl.h>
#include <libintl.h>
#include <locale.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <error.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <ctype.h>

#include "localedef.h"
#include "charmap.h"
#include "locfile.h"

/* Undefine the following line in the production version.  */
/* #define NDEBUG 1 */
#include <assert.h>


/* List of copied locales.  */
struct copy_def_list_t *copy_list;

/* If this is defined be POSIX conform.  */
int posix_conformance;

/* If not zero force output even if warning were issued.  */
static int force_output;

/* Prefix for output files.  */
const char *output_prefix;

/* Name of the character map file.  */
static const char *charmap_file;

/* Name of the locale definition file.  */
static const char *input_file;

/* Name of the repertoire map file.  */
const char *repertoire_global;

/* Name of the locale.alias file.  */
const char *alias_file;

/* List of all locales.  */
static struct localedef_t *locales;

/* If true don't add locale data to archive.  */
bool no_archive;

/* If true add named locales to archive.  */
static bool add_to_archive;

/* If true delete named locales from archive.  */
static bool delete_from_archive;

/* If true replace archive content when adding.  */
static bool replace_archive;

/* If true list archive content.  */
static bool list_archive;

/* If true create hard links to other locales (default).  */
bool hard_links = true;

/* Maximum number of retries when opening the locale archive.  */
int max_locarchive_open_retry = 10;


/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

#define OPT_POSIX 301
#define OPT_QUIET 302
#define OPT_PREFIX 304
#define OPT_NO_ARCHIVE 305
#define OPT_ADD_TO_ARCHIVE 306
#define OPT_REPLACE 307
#define OPT_DELETE_FROM_ARCHIVE 308
#define OPT_LIST_ARCHIVE 309
#define OPT_LITTLE_ENDIAN 400
#define OPT_BIG_ENDIAN 401
#define OPT_NO_WARN 402
#define OPT_WARN 403
#define OPT_NO_HARD_LINKS 404

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { NULL, 0, NULL, 0, N_("Input Files:") },
  { "charmap", 'f', N_("FILE"), 0,
    N_("Symbolic character names defined in FILE") },
  { "inputfile", 'i', N_("FILE"), 0,
    N_("Source definitions are found in FILE") },
  { "repertoire-map", 'u', N_("FILE"), 0,
    N_("FILE contains mapping from symbolic names to UCS4 values") },

  { NULL, 0, NULL, 0, N_("Output control:") },
  { "force", 'c', NULL, 0,
    N_("Create output even if warning messages were issued") },
  { "no-hard-links", OPT_NO_HARD_LINKS, NULL, 0,
    N_("Do not create hard links between installed locales") },
  { "prefix", OPT_PREFIX, N_("PATH"), 0, N_("Optional output file prefix") },
  { "posix", OPT_POSIX, NULL, 0, N_("Strictly conform to POSIX") },
  { "quiet", OPT_QUIET, NULL, 0,
    N_("Suppress warnings and information messages") },
  { "verbose", 'v', NULL, 0, N_("Print more messages") },
  { "no-warnings", OPT_NO_WARN, N_("<warnings>"), 0,
    N_("Comma-separated list of warnings to disable; "
       "supported warnings are: ascii, intcurrsym") },
  { "warnings", OPT_WARN, N_("<warnings>"), 0,
    N_("Comma-separated list of warnings to enable; "
       "supported warnings are: ascii, intcurrsym") },

  { NULL, 0, NULL, 0, N_("Archive control:") },
  { "no-archive", OPT_NO_ARCHIVE, NULL, 0,
    N_("Don't add new data to archive") },
  { "add-to-archive", OPT_ADD_TO_ARCHIVE, NULL, 0,
    N_("Add locales named by parameters to archive") },
  { "replace", OPT_REPLACE, NULL, 0, N_("Replace existing archive content") },
  { "delete-from-archive", OPT_DELETE_FROM_ARCHIVE, NULL, 0,
    N_("Remove locales named by parameters from archive") },
  { "list-archive", OPT_LIST_ARCHIVE, NULL, 0, N_("List content of archive") },
  { "alias-file", 'A', N_("FILE"), 0,
    N_("locale.alias file to consult when making archive")},
  { "little-endian", OPT_LITTLE_ENDIAN, NULL, 0,
    N_("Generate little-endian output") },
  { "big-endian", OPT_BIG_ENDIAN, NULL, 0,
    N_("Generate big-endian output") },
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("Compile locale specification");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("\
NAME\n\
[--add-to-archive|--delete-from-archive] FILE...\n\
--list-archive [FILE]");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help
};


/* Prototypes for local functions.  */
static void error_print (void);
static char *construct_output_path (char *path);
static char *normalize_codeset (const char *codeset, size_t name_len);


int
main (int argc, char *argv[])
{
  char *output_path;
  int cannot_write_why;
  struct charmap_t *charmap;
  struct localedef_t global;
  int remaining;

  /* Set initial values for global variables.  */
  copy_list = NULL;
  posix_conformance = getenv ("POSIXLY_CORRECT") != NULL;
  error_print_progname = error_print;

  /* Set locale.  Do not set LC_ALL because the other categories must
     not be affected (according to POSIX.2).  */
  setlocale (LC_MESSAGES, "");
  setlocale (LC_CTYPE, "");

  /* Initialize the message catalog.  */
  textdomain (_libc_intl_domainname);

  /* Parse and process arguments.  */
  argp_err_exit_status = 4;
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  /* Handle a few special cases.  */
  if (list_archive)
    show_archive_content (remaining > 1 ? argv[remaining] : NULL, verbose);
  if (add_to_archive)
    return add_locales_to_archive (argc - remaining, &argv[remaining],
				   replace_archive);
  if (delete_from_archive)
    return delete_locales_from_archive (argc - remaining, &argv[remaining]);

  /* POSIX.2 requires to be verbose about missing characters in the
     character map.  */
  verbose |= posix_conformance;

  if (argc - remaining != 1)
    {
      /* We need exactly one non-option parameter.  */
      argp_help (&argp, stdout, ARGP_HELP_SEE | ARGP_HELP_EXIT_ERR,
		 program_invocation_short_name);
      exit (4);
    }

  /* The parameter describes the output path of the constructed files.
     If the described files cannot be written return a NULL pointer.
     We don't free output_path because we will exit.  */
  output_path  = construct_output_path (argv[remaining]);
  if (output_path == NULL && ! no_archive)
    error (4, errno, _("cannot create directory for output files"));
  cannot_write_why = errno;

  /* Now that the parameters are processed we have to reset the local
     ctype locale.  (P1003.2 4.35.5.2)  */
  setlocale (LC_CTYPE, "POSIX");

  /* Look whether the system really allows locale definitions.  POSIX
     defines error code 3 for this situation so I think it must be
     a fatal error (see P1003.2 4.35.8).  */
  if (sysconf (_SC_2_LOCALEDEF) < 0)
    record_error (3, 0, _("\
FATAL: system does not define `_POSIX2_LOCALEDEF'"));

  /* Process charmap file.  */
  charmap = charmap_read (charmap_file, verbose, 1, be_quiet, 1);

  /* Add the first entry in the locale list.  */
  memset (&global, '\0', sizeof (struct localedef_t));
  global.name = input_file ?: "/dev/stdin";
  global.needed = ALL_LOCALES;
  locales = &global;

  /* Now read the locale file.  */
  if (locfile_read (&global, charmap) != 0)
    record_error (4, errno, _("\
cannot open locale definition file `%s'"), input_file);

  /* Perhaps we saw some `copy' instructions.  */
  while (1)
    {
      struct localedef_t *runp = locales;

      while (runp != NULL && (runp->needed & runp->avail) == runp->needed)
	runp = runp->next;

      if (runp == NULL)
	/* Everything read.  */
	break;

      if (locfile_read (runp, charmap) != 0)
	record_error (4, errno, _("\
cannot open locale definition file `%s'"), runp->name);
    }

  /* Check the categories we processed in source form.  */
  check_all_categories (locales, charmap);

  /* What we do next depends on the number of errors and warnings we
     have generated in processing the input files.

     * No errors: Write the output file.

     * Some warnings: Write the output file and exit with status 1 to
     indicate there may be problems using the output file e.g. missing
     data that makes it difficult to use

     * Errors: We don't write the output file and we exit with status 4
     to indicate no output files were written.

     The use of -c|--force writes the output file even if errors were
     seen.  */
  if (recorded_error_count == 0 || force_output != 0)
    {
      if (cannot_write_why != 0)
	record_error (4, cannot_write_why, _("\
cannot write output files to `%s'"), output_path ? : argv[remaining]);
      else
	write_all_categories (locales, charmap, argv[remaining], output_path);
    }
  else
    record_error (4, 0, _("\
no output file produced because errors were issued"));

  /* This exit status is prescribed by POSIX.2 4.35.7.  */
  exit (recorded_warning_count != 0);
}

/* Search warnings for matching warnings and if found enable those
   warnings if ENABLED is true, otherwise disable the warnings.  */
static void
set_warnings (char *warnings, bool enabled)
{
  char *tok = warnings;
  char *copy = (char *) malloc (strlen (warnings) + 1);
  char *save = copy;

  /* As we make a copy of the warnings list we remove all spaces from
     the warnings list to make the processing a more robust.  We don't
     support spaces in a warning name.  */
  do
    {
      while (isspace (*tok) != 0)
        tok++;
    }
  while ((*save++ = *tok++) != '\0');

  warnings = copy;

  /* Tokenize the input list of warnings to set, compare them to
     known warnings, and set the warning.  We purposely ignore unknown
     warnings, and are thus forward compatible, users can attempt to
     disable whaterver new warnings they know about, but we will only
     disable those *we* known about.  */
  while ((tok = strtok_r (warnings, ",", &save)) != NULL)
    {
      warnings = NULL;
      if (strcmp (tok, "ascii") == 0)
	warn_ascii = enabled;
      else if (strcmp (tok, "intcurrsym") == 0)
	warn_int_curr_symbol = enabled;
    }

  free (copy);
}

/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case OPT_QUIET:
      be_quiet = 1;
      break;
    case OPT_POSIX:
      posix_conformance = 1;
      break;
    case OPT_PREFIX:
      output_prefix = arg;
      break;
    case OPT_NO_ARCHIVE:
      no_archive = true;
      break;
    case OPT_ADD_TO_ARCHIVE:
      add_to_archive = true;
      break;
    case OPT_REPLACE:
      replace_archive = true;
      break;
    case OPT_DELETE_FROM_ARCHIVE:
      delete_from_archive = true;
      break;
    case OPT_LIST_ARCHIVE:
      list_archive = true;
      break;
    case OPT_LITTLE_ENDIAN:
      set_big_endian (false);
      break;
    case OPT_BIG_ENDIAN:
      set_big_endian (true);
      break;
    case OPT_NO_WARN:
      /* Disable the warnings.  */
      set_warnings (arg, false);
      break;
    case OPT_WARN:
      /* Enable the warnings.  */
      set_warnings (arg, true);
      break;
    case OPT_NO_HARD_LINKS:
      /* Do not hard link to other locales.  */
      hard_links = false;
      break;
    case 'c':
      force_output = 1;
      break;
    case 'f':
      charmap_file = arg;
      break;
    case 'A':
      alias_file = arg;
      break;
    case 'i':
      input_file = arg;
      break;
    case 'u':
      repertoire_global = arg;
      break;
    case 'v':
      verbose = 1;
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


static char *
more_help (int key, const char *text, void *input)
{
  char *cp;
  char *tp;

  switch (key)
    {
    case ARGP_KEY_HELP_EXTRA:
      /* We print some extra information.  */
      tp = xasprintf (gettext ("\
For bug reporting instructions, please see:\n\
%s.\n"), REPORT_BUGS_TO);
      cp = xasprintf (gettext ("\
System's directory for character maps : %s\n\
		       repertoire maps: %s\n\
		       locale path    : %s\n\
%s"),
		    CHARMAP_PATH, REPERTOIREMAP_PATH, LOCALE_PATH, tp);
      free (tp);
      return cp;
    default:
      break;
    }
  return (char *) text;
}

/* Print the version information.  */
static void
print_version (FILE *stream, struct argp_state *state)
{
  fprintf (stream, "localedef %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


/* The address of this function will be assigned to the hook in the error
   functions.  */
static void
error_print (void)
{
}


/* The parameter to localedef describes the output path.  If it does contain a
   '/' character it is a relative path.  Otherwise it names the locale this
   definition is for.   The returned path must be freed by the caller. */
static char *
construct_output_path (char *path)
{
  char *result;

  if (strchr (path, '/') == NULL)
    {
      /* This is a system path.  First examine whether the locale name
	 contains a reference to the codeset.  This should be
	 normalized.  */
      char *startp;
      char *endp = NULL;
      char *normal = NULL;

      startp = path;
      /* Either we have a '@' which starts a CEN name or '.' which starts the
	 codeset specification.  The CEN name starts with '@' and may also have
	 a codeset specification, but we do not normalize the string after '@'.
	 If we only find the codeset specification then we normalize only the codeset
	 specification (but not anything after a subsequent '@').  */
      while (*startp != '\0' && *startp != '@' && *startp != '.')
	++startp;
      if (*startp == '.')
	{
	  /* We found a codeset specification.  Now find the end.  */
	  endp = ++startp;

	  /* Stop at the first '@', and don't normalize anything past that.  */
	  while (*endp != '\0' && *endp != '@')
	    ++endp;

	  if (endp > startp)
	    normal = normalize_codeset (startp, endp - startp);
	}

      if (normal == NULL)
	result = xasprintf ("%s%s/%s/", output_prefix ?: "",
			    COMPLOCALEDIR, path);
      else
	result = xasprintf ("%s%s/%.*s%s%s/",
			    output_prefix ?: "", COMPLOCALEDIR,
			    (int) (startp - path), path, normal, endp ?: "");
      /* Free the allocated normalized codeset name.  */
      free (normal);
    }
  else
    {
      /* This is a user path.  */
      result = xasprintf ("%s/", path);

      /* If the user specified an output path we cannot add the output
	 to the archive.  */
      no_archive = true;
    }

  errno = 0;

  if (no_archive && euidaccess (result, W_OK) == -1)
    {
      /* Perhaps the directory does not exist now.  Try to create it.  */
      if (errno == ENOENT)
	{
	  errno = 0;
	  if (mkdir (result, 0777) < 0)
	    {
	      record_verbose (stderr,
			      _("cannot create output path \'%s\': %s"),
			      result, strerror (errno));
	      free (result);
	      return NULL;
	    }
	}
      else
	record_verbose (stderr,
			_("no write permission to output path \'%s\': %s"),
			result, strerror (errno));
    }

  return result;
}


/* Normalize codeset name.  There is no standard for the codeset names.
   Normalization allows the user to use any of the common names e.g. UTF-8,
   utf-8, utf8, UTF8 etc.

   We normalize using the following rules:
   - Remove all non-alpha-numeric characters
   - Lowercase all characters.
   - If there are only digits assume it's an ISO standard and prefix with 'iso'

   We return the normalized string which needs to be freed by free.  */
static char *
normalize_codeset (const char *codeset, size_t name_len)
{
  int len = 0;
  int only_digit = 1;
  char *retval;
  char *wp;
  size_t cnt;

  /* Compute the length of only the alpha-numeric characters.  */
  for (cnt = 0; cnt < name_len; ++cnt)
    if (isalnum (codeset[cnt]))
      {
	++len;

	if (isalpha (codeset[cnt]))
	  only_digit = 0;
      }

  /* If there were only digits we assume it's an ISO standard and we will
     prefix with 'iso' so include space for that.  We fill in the required
     space from codeset up to the converted length.  */
  wp = retval = xasprintf ("%s%.*s", only_digit ? "iso" : "", len, codeset);

  /* Skip "iso".  */
  if (only_digit)
    wp += 3;

  /* Lowercase all characters. */
  for (cnt = 0; cnt < name_len; ++cnt)
    if (isalpha (codeset[cnt]))
      *wp++ = tolower (codeset[cnt]);
    else if (isdigit (codeset[cnt]))
      *wp++ = codeset[cnt];

  /* Return allocated and converted name for caller to free.  */
  return retval;
}


struct localedef_t *
add_to_readlist (int category, const char *name, const char *repertoire_name,
		 int generate, struct localedef_t *copy_locale)
{
  struct localedef_t *runp = locales;

  while (runp != NULL && strcmp (name, runp->name) != 0)
    runp = runp->next;

  if (runp == NULL)
    {
      /* Add a new entry at the end.  */
      struct localedef_t *newp;

      assert (generate == 1);

      newp = xcalloc (1, sizeof (struct localedef_t));
      newp->name = name;
      newp->repertoire_name = repertoire_name;

      if (locales == NULL)
	runp = locales = newp;
      else
	{
	  runp = locales;
	  while (runp->next != NULL)
	    runp = runp->next;
	  runp = runp->next = newp;
	}
    }

  if (generate
      && (runp->needed & (1 << category)) != 0
      && (runp->avail & (1 << category)) == 0)
    record_error (5, 0, _("\
circular dependencies between locale definitions"));

  if (copy_locale != NULL)
    {
      if (runp->categories[category].generic != NULL)
	record_error (5, 0, _("\
cannot add already read locale `%s' a second time"), name);
      else
	runp->categories[category].generic =
	  copy_locale->categories[category].generic;
    }

  runp->needed |= 1 << category;

  return runp;
}


struct localedef_t *
find_locale (int category, const char *name, const char *repertoire_name,
	     const struct charmap_t *charmap)
{
  struct localedef_t *result;

  /* Find the locale, but do not generate it since this would be a bug.  */
  result = add_to_readlist (category, name, repertoire_name, 0, NULL);

  assert (result != NULL);

  if ((result->avail & (1 << category)) == 0
      && locfile_read (result, charmap) != 0)
    record_error (4, errno, _("\
cannot open locale definition file `%s'"), result->name);

  return result;
}


struct localedef_t *
load_locale (int category, const char *name, const char *repertoire_name,
	     const struct charmap_t *charmap, struct localedef_t *copy_locale)
{
  struct localedef_t *result;

  /* Generate the locale if it does not exist.  */
  result = add_to_readlist (category, name, repertoire_name, 1, copy_locale);

  assert (result != NULL);

  if ((result->avail & (1 << category)) == 0
      && locfile_read (result, charmap) != 0)
    record_error (4, errno, _("\
cannot open locale definition file `%s'"), result->name);

  return result;
}
