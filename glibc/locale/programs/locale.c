/* Implementation of the locale program according to POSIX 9945-2.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#include <argz.h>
#include <dirent.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <langinfo.h>
#include <libintl.h>
#include <limits.h>
#include <locale.h>
#include <search.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "record-status.h"
#include "localeinfo.h"
#include "charmap-dir.h"
#include "../locarchive.h"
#include <programs/xmalloc.h>

#define ARCHIVE_NAME COMPLOCALEDIR "/locale-archive"

/* If set print the name of the category.  */
static int show_category_name;

/* If set print the name of the item.  */
static int show_keyword_name;

/* Print names of all available locales.  */
static int do_all;

/* Print names of all available character maps.  */
static int do_charmaps = 0;

/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { NULL, 0, NULL, 0, N_("System information:") },
  { "all-locales", 'a', NULL, OPTION_NO_USAGE,
    N_("Write names of available locales") },
  { "charmaps", 'm', NULL, OPTION_NO_USAGE,
    N_("Write names of available charmaps") },
  { NULL, 0, NULL, 0, N_("Modify output format:") },
  { "category-name", 'c', NULL, 0, N_("Write names of selected categories") },
  { "keyword-name", 'k', NULL, 0, N_("Write names of selected keywords") },
  { "verbose", 'v', NULL, 0, N_("Print more information") },
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("Get locale-specific information.");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("NAME\n[-a|-m]");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help
};


/* We don't have these constants defined because we don't use them.  Give
   default values.  */
#define CTYPE_MB_CUR_MIN 0
#define CTYPE_MB_CUR_MAX 0
#define CTYPE_HASH_SIZE 0
#define CTYPE_HASH_LAYERS 0
#define CTYPE_CLASS 0
#define CTYPE_TOUPPER_EB 0
#define CTYPE_TOLOWER_EB 0
#define CTYPE_TOUPPER_EL 0
#define CTYPE_TOLOWER_EL 0

/* Definition of the data structure which represents a category and its
   items.  */
struct category
{
  int cat_id;
  const char *name;
  size_t number;
  struct cat_item
  {
    int item_id;
    const char *name;
    enum { std, opt } status;
    enum value_type value_type;
    int min;
    int max;
  } *item_desc;
};

/* Simple helper macro.  */
#define NELEMS(arr) ((sizeof (arr)) / (sizeof (arr[0])))

/* For some tricky stuff.  */
#define NO_PAREN(Item, More...) Item, ## More

/* We have all categories defined in `categories.def'.  Now construct
   the description and data structure used for all categories.  */
#define DEFINE_ELEMENT(Item, More...) { Item, ## More },
#define DEFINE_CATEGORY(category, name, items, postload) \
    static struct cat_item category##_desc[] =				      \
      {									      \
	NO_PAREN items							      \
      };

#include "categories.def"
#undef DEFINE_CATEGORY

static struct category category[] =
  {
#define DEFINE_CATEGORY(category, name, items, postload) \
    [category] = { _NL_NUM_##category, name, NELEMS (category##_desc),	      \
		   category##_desc },
#include "categories.def"
#undef DEFINE_CATEGORY
  };
#define NCATEGORIES NELEMS (category)


/* Automatically set variable.  */
extern const char *__progname;

/* helper function for extended name handling.  */
extern void locale_special (const char *name, int show_category_name,
			    int show_keyword_name);

/* Prototypes for local functions.  */
static void print_LC_IDENTIFICATION (void *mapped, size_t size);
static void print_LC_CTYPE (void *mapped, size_t size);
static void write_locales (void);
static int nameentcmp (const void *a, const void *b);
static int write_archive_locales (void **all_datap, char *linebuf);
static void write_charmaps (void);
static void show_locale_vars (void);
static void show_info (const char *name);
static void try_setlocale (int category, const char *category_name);
static char *quote_string (const char *input);
static void setlocale_diagnostics (void);


int
main (int argc, char *argv[])
{
  int remaining;

  /* Set initial values for global variables.  */
  show_category_name = 0;
  show_keyword_name = 0;

  /* Set locale.  Do not set LC_ALL because the other categories must
     not be affected (according to POSIX.2).  */
  try_setlocale (LC_CTYPE, "LC_CTYPE");
  try_setlocale (LC_MESSAGES, "LC_MESSAGES");

  /* Initialize the message catalog.  */
  textdomain (PACKAGE);

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  /* `-a' requests the names of all available locales.  */
  if (do_all != 0)
    {
      setlocale_diagnostics ();
      try_setlocale (LC_COLLATE, "LC_COLLATE");
      write_locales ();
      exit (EXIT_SUCCESS);
    }

  /* `m' requests the names of all available charmaps.  The names can be
     used for the -f argument to localedef(1).  */
  if (do_charmaps != 0)
    {
      setlocale_diagnostics ();
      write_charmaps ();
      exit (EXIT_SUCCESS);
    }

  /* Specific information about the current locale are requested.
     Change to this locale now.  */
  try_setlocale (LC_ALL, "LC_ALL");
  setlocale_diagnostics ();

  /* If no real argument is given we have to print the contents of the
     current locale definition variables.  These are LANG and the LC_*.  */
  if (remaining == argc && show_keyword_name == 0 && show_category_name == 0)
    {
      show_locale_vars ();
      exit (EXIT_SUCCESS);
    }

  /* Process all given names.  */
  while (remaining <  argc)
    show_info (argv[remaining++]);

  exit (EXIT_SUCCESS);
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'a':
      do_all = 1;
      break;
    case 'c':
      show_category_name = 1;
      break;
    case 'm':
      do_charmaps = 1;
      break;
    case 'k':
      show_keyword_name = 1;
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
  fprintf (stream, "locale %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


/* Simple action function which prints arguments as strings.  */
static void
print_names (const void *nodep, VISIT value, int level)
{
  if (value == postorder || value == leaf)
    puts (*(char **) nodep);
}


static int
select_dirs (const struct dirent *dirent)
{
  int result = 0;

  if (strcmp (dirent->d_name, ".") != 0 && strcmp (dirent->d_name, "..") != 0)
    {
      mode_t mode = 0;

      if (dirent->d_type != DT_UNKNOWN && dirent->d_type != DT_LNK)
	mode = DTTOIF (dirent->d_type);
      else
	{
	  struct stat64 st;
	  char buf[sizeof (COMPLOCALEDIR)
		   + strlen (dirent->d_name) + 1];

	  stpcpy (stpcpy (stpcpy (buf, COMPLOCALEDIR), "/"),
		  dirent->d_name);

	  if (stat64 (buf, &st) == 0)
	    mode = st.st_mode;
	}

      result = S_ISDIR (mode);
    }

  return result;
}


static void
print_LC_IDENTIFICATION (void *mapped, size_t size)
{
  /* Read the information from the file.  */
  struct
    {
      unsigned int magic;
      unsigned int nstrings;
      unsigned int strindex[0];
    } *filedata = mapped;

  if (filedata->magic == LIMAGIC (LC_IDENTIFICATION)
      && (sizeof *filedata
	  + (filedata->nstrings
	     * sizeof (unsigned int))
	  <= size))
    {
      const char *str;

#define HANDLE(idx, name) \
  str = ((char *) mapped						      \
	 + filedata->strindex[_NL_ITEM_INDEX (_NL_IDENTIFICATION_##idx)]);    \
  if (*str != '\0')							      \
    printf ("%9s | %s\n", name, str)
      HANDLE (TITLE, "title");
      HANDLE (SOURCE, "source");
      HANDLE (ADDRESS, "address");
      HANDLE (CONTACT, "contact");
      HANDLE (EMAIL, "email");
      HANDLE (TEL, "telephone");
      HANDLE (FAX, "fax");
      HANDLE (LANGUAGE, "language");
      HANDLE (TERRITORY, "territory");
      HANDLE (AUDIENCE, "audience");
      HANDLE (APPLICATION, "application");
      HANDLE (ABBREVIATION, "abbreviation");
      HANDLE (REVISION, "revision");
      HANDLE (DATE, "date");
    }
}


static void
print_LC_CTYPE (void *mapped, size_t size)
{
  struct
    {
      unsigned int magic;
      unsigned int nstrings;
      unsigned int strindex[0];
    } *filedata = mapped;

  if (filedata->magic == LIMAGIC (LC_CTYPE)
      && (sizeof *filedata
	  + (filedata->nstrings
	     * sizeof (unsigned int))
	  <= size))
    {
      const char *str;

      str = ((char *) mapped
	     + filedata->strindex[_NL_ITEM_INDEX (_NL_CTYPE_CODESET_NAME)]);
      if (*str != '\0')
	printf ("  codeset | %s\n", str);
    }
}


/* Write the names of all available locales to stdout.  We have some
   sources of the information: the contents of the locale directory
   and the locale.alias file.  To avoid duplicates and print the
   result is a reasonable order we put all entries is a search tree
   and print them afterwards.  */
static void
write_locales (void)
{
  char linebuf[80];
  void *all_data = NULL;
  struct dirent **dirents;
  int ndirents;
  int cnt;
  char *alias_path;
  size_t alias_path_len;
  char *entry;
  int first_locale = 1;

#define PUT(name) tsearch (name, &all_data, \
			   (int (*) (const void *, const void *)) strcoll)
#define GET(name) tfind (name, &all_data, \
			   (int (*) (const void *, const void *)) strcoll)

  /* `POSIX' locale is always available (POSIX.2 4.34.3).  */
  PUT ("POSIX");
  /* And so is the "C" locale.  */
  PUT ("C");

  memset (linebuf, '-', sizeof (linebuf) - 1);
  linebuf[sizeof (linebuf) - 1] = '\0';

  /* First scan the locale archive.  */
  if (write_archive_locales (&all_data, linebuf))
    first_locale = 0;

  /* Now we can look for all files in the directory.  */
  ndirents = scandir (COMPLOCALEDIR, &dirents, select_dirs,
		      alphasort);
  for (cnt = 0; cnt < ndirents; ++cnt)
    {
      /* Test whether at least the LC_CTYPE data is there.  Some
	 directories only contain translations.  */
      char buf[sizeof (COMPLOCALEDIR)
	       + strlen (dirents[cnt]->d_name)
	       + sizeof "/LC_IDENTIFICATION"];
      char *enddir;
      struct stat64 st;

      stpcpy (enddir = stpcpy (stpcpy (stpcpy (buf,
					       COMPLOCALEDIR),
					       "/"),
			       dirents[cnt]->d_name),
	      "/LC_IDENTIFICATION");

      if (stat64 (buf, &st) == 0 && S_ISREG (st.st_mode))
	{
	  if (verbose && GET (dirents[cnt]->d_name) == NULL)
	    {
	      /* Provide some nice output of all kinds of
		 information.  */
	      int fd;

	      if (! first_locale)
		putchar_unlocked ('\n');
	      first_locale = 0;

	      printf ("locale: %-15.15s directory: %.*s\n%s\n",
		      dirents[cnt]->d_name, (int) (enddir - buf), buf,
		      linebuf);

	      fd = open64 (buf, O_RDONLY);
	      if (fd != -1)
		{
		  void *mapped = mmap64 (NULL, st.st_size, PROT_READ,
					 MAP_SHARED, fd, 0);
		  if (mapped != MAP_FAILED)
		    {
		      print_LC_IDENTIFICATION (mapped, st.st_size);

		      munmap (mapped, st.st_size);
		    }

		  close (fd);

		  /* Now try to get the charset information.  */
		  strcpy (enddir, "/LC_CTYPE");
		  fd = open64 (buf, O_RDONLY);
		  if (fd != -1 && fstat64 (fd, &st) >= 0
		      && ((mapped = mmap64 (NULL, st.st_size, PROT_READ,
					    MAP_SHARED, fd, 0))
			  != MAP_FAILED))
		    {
		      print_LC_CTYPE (mapped, st.st_size);

		      munmap (mapped, st.st_size);
		    }

		  if (fd != -1)
		    close (fd);
		}
	    }

	  /* If the verbose format is not selected we simply
	     collect the names.  */
	  PUT (xstrdup (dirents[cnt]->d_name));
	}
    }
  if (ndirents > 0)
    free (dirents);

  /* Now read the locale.alias files.  */
  if (argz_create_sep (LOCALE_ALIAS_PATH, ':', &alias_path, &alias_path_len))
    error (1, errno, gettext ("while preparing output"));

  entry = NULL;
  while ((entry = argz_next (alias_path, alias_path_len, entry)))
    {
      static const char aliasfile[] = "/locale.alias";
      FILE *fp;
      char full_name[strlen (entry) + sizeof aliasfile];

      stpcpy (stpcpy (full_name, entry), aliasfile);
      fp = fopen (full_name, "rm");
      if (fp == NULL)
	/* Ignore non-existing files.  */
	continue;

      /* No threads present.  */
      __fsetlocking (fp, FSETLOCKING_BYCALLER);

      while (! feof_unlocked (fp))
	{
	  /* It is a reasonable approach to use a fix buffer here
	     because
	     a) we are only interested in the first two fields
	     b) these fields must be usable as file names and so must
		not be that long  */
	  char buf[BUFSIZ];
	  char *alias;
	  char *value;
	  char *cp;

	  if (fgets_unlocked (buf, BUFSIZ, fp) == NULL)
	    /* EOF reached.  */
	    break;

	  cp = buf;
	  /* Ignore leading white space.  */
	  while (isspace (cp[0]) && cp[0] != '\n')
	    ++cp;

	  /* A leading '#' signals a comment line.  */
	  if (cp[0] != '\0' && cp[0] != '#' && cp[0] != '\n')
	    {
	      alias = cp++;
	      while (cp[0] != '\0' && !isspace (cp[0]))
		++cp;
	      /* Terminate alias name.  */
	      if (cp[0] != '\0')
		*cp++ = '\0';

	      /* Now look for the beginning of the value.  */
	      while (isspace (cp[0]))
		++cp;

	      if (cp[0] != '\0')
		{
		  value = cp++;
		  while (cp[0] != '\0' && !isspace (cp[0]))
		    ++cp;
		  /* Terminate value.  */
		  if (cp[0] == '\n')
		    {
		      /* This has to be done to make the following
			 test for the end of line possible.  We are
			 looking for the terminating '\n' which do not
			 overwrite here.  */
		      *cp++ = '\0';
		      *cp = '\n';
		    }
		  else if (cp[0] != '\0')
		    *cp++ = '\0';

		  /* Add the alias.  */
		  if (! verbose && GET (value) != NULL)
		    PUT (xstrdup (alias));
		}
	    }

	  /* Possibly not the whole line fits into the buffer.
	     Ignore the rest of the line.  */
	  while (strchr (cp, '\n') == NULL)
	    {
	      cp = buf;
	      if (fgets_unlocked (buf, BUFSIZ, fp) == NULL)
		/* Make sure the inner loop will be left.  The outer
		   loop will exit at the `feof' test.  */
		*cp = '\n';
	    }
	}

      fclose (fp);
    }

  if (! verbose)
    {
      twalk (all_data, print_names);
    }
}


struct nameent
{
  char *name;
  uint32_t locrec_offset;
};


static int
nameentcmp (const void *a, const void *b)
{
  return strcoll (((const struct nameent *) a)->name,
		  ((const struct nameent *) b)->name);
}


static int
write_archive_locales (void **all_datap, char *linebuf)
{
  struct stat64 st;
  void *all_data = *all_datap;
  size_t len = 0;
  struct locarhead *head;
  struct namehashent *namehashtab;
  char *addr = MAP_FAILED;
  int fd, ret = 0;
  uint32_t cnt;

  fd = open64 (ARCHIVE_NAME, O_RDONLY);
  if (fd < 0)
    return 0;

  if (fstat64 (fd, &st) < 0 || st.st_size < sizeof (*head))
    goto error_out;

  len = st.st_size;
  addr = mmap64 (NULL, len, PROT_READ, MAP_SHARED, fd, 0);
  if (addr == MAP_FAILED)
    goto error_out;

  head = (struct locarhead *) addr;
  if (head->namehash_offset + head->namehash_size > len
      || head->string_offset + head->string_size > len
      || head->locrectab_offset + head->locrectab_size > len
      || head->sumhash_offset + head->sumhash_size > len)
    goto error_out;

  namehashtab = (struct namehashent *) (addr + head->namehash_offset);
  if (! verbose)
    {
      for (cnt = 0; cnt < head->namehash_size; ++cnt)
	if (namehashtab[cnt].locrec_offset != 0)
	  {
	    PUT (xstrdup (addr + namehashtab[cnt].name_offset));
	    ++ret;
	  }
    }
  else
    {
      struct nameent *names;
      uint32_t used;

      names = (struct nameent *) xmalloc (head->namehash_used
					  * sizeof (struct nameent));
      for (cnt = used = 0; cnt < head->namehash_size; ++cnt)
	if (namehashtab[cnt].locrec_offset != 0)
	  {
	    names[used].name = addr + namehashtab[cnt].name_offset;
	    names[used++].locrec_offset = namehashtab[cnt].locrec_offset;
	  }

      /* Sort the names.  */
      qsort (names, used, sizeof (struct nameent), nameentcmp);

      for (cnt = 0; cnt < used; ++cnt)
	{
	  struct locrecent *locrec;

	  PUT (xstrdup (names[cnt].name));

	  if (cnt)
	    putchar_unlocked ('\n');

	  printf ("locale: %-15.15s archive: " ARCHIVE_NAME "\n%s\n",
		  names[cnt].name, linebuf);

	  locrec = (struct locrecent *) (addr + names[cnt].locrec_offset);

	  print_LC_IDENTIFICATION (addr
				   + locrec->record[LC_IDENTIFICATION].offset,
				   locrec->record[LC_IDENTIFICATION].len);

	  print_LC_CTYPE (addr + locrec->record[LC_CTYPE].offset,
			  locrec->record[LC_CTYPE].len);
	}

      free (names);

      ret = used;
    }

error_out:
  if (addr != MAP_FAILED)
    munmap (addr, len);
  close (fd);
  *all_datap = all_data;
  return ret;
}


/* Write the names of all available character maps to stdout.  */
static void
write_charmaps (void)
{
  void *all_data = NULL;
  CHARMAP_DIR *dir;
  const char *dirent;

  /* Look for all files in the charmap directory.  */
  dir = charmap_opendir (CHARMAP_PATH);
  if (dir == NULL)
    return;

  while ((dirent = charmap_readdir (dir)) != NULL)
    {
      char **aliases;
      char **p;

      PUT (xstrdup (dirent));

      aliases = charmap_aliases (CHARMAP_PATH, dirent);

#if 0
      /* Add the code_set_name and the aliases.  */
      for (p = aliases; *p; p++)
	PUT (xstrdup (*p));
#else
      /* Add the code_set_name only.  Most aliases are obsolete.  */
      p = aliases;
      if (*p)
	PUT (xstrdup (*p));
#endif

      charmap_free_aliases (aliases);
    }

  charmap_closedir (dir);

  twalk (all_data, print_names);
}

/* Print a properly quoted assignment of NAME with VAL, using double
   quotes iff DQUOTE is true.  */
static void
print_assignment (const char *name, const char *val, bool dquote)
{
  printf ("%s=", name);
  if (dquote)
    putchar ('"');
  while (*val != '\0')
    {
      size_t segment
	= strcspn (val, dquote ? "$`\"\\" : "~|&;<>()$`\\\"' \t\n");
      printf ("%.*s", (int) segment, val);
      val += segment;
      if (*val == '\0')
	break;
      putchar ('\\');
      putchar (*val++);
    }
  if (dquote)
    putchar ('"');
  putchar ('\n');
}

/* We have to show the contents of the environments determining the
   locale.  */
static void
show_locale_vars (void)
{
  const char *lcall = getenv ("LC_ALL") ?: "";
  const char *lang = getenv ("LANG") ?: "";

  /* LANG has to be the first value.  */
  print_assignment ("LANG", lang, false);

  /* Now all categories in an unspecified order.  */
  for (size_t cat_no = 0; cat_no < NCATEGORIES; ++cat_no)
    if (cat_no != LC_ALL)
      {
	const char *name = category[cat_no].name;
	const char *val = getenv (name);

	if (lcall[0] != '\0' || val == NULL)
	  print_assignment (name,
			    lcall[0] != '\0' ? lcall
			    : lang[0] != '\0' ? lang
			    : "POSIX",
			    true);
	else
	  print_assignment (name, val, false);
      }

  /* The last is the LC_ALL value.  */
  print_assignment ("LC_ALL", lcall, false);
}


/* Subroutine of show_info, below.  */
static void
print_item (struct cat_item *item)
{
  switch (item->value_type)
    {
    case string:
      if (show_keyword_name)
	printf ("%s=\"", item->name);
      fputs (nl_langinfo (item->item_id) ? : "", stdout);
      if (show_keyword_name)
	putchar ('"');
      putchar ('\n');
      break;
    case stringarray:
      {
	const char *val;
	int cnt;

	if (show_keyword_name)
	  printf ("%s=\"", item->name);

	for (cnt = 0; cnt < item->max - 1; ++cnt)
	  {
	    val = nl_langinfo (item->item_id + cnt);
	    if (val != NULL)
	      fputs (val, stdout);
	    putchar (';');
	  }

	val = nl_langinfo (item->item_id + cnt);
	if (val != NULL)
	  fputs (val, stdout);

	if (show_keyword_name)
	  putchar ('"');
	putchar ('\n');
      }
      break;
    case stringlist:
      {
	int first = 1;
	const char *val = nl_langinfo (item->item_id) ? : "";

	if (show_keyword_name)
	  printf ("%s=", item->name);

	for (int cnt = 0; cnt < item->max && *val != '\0'; ++cnt)
	  {
	    printf ("%s%s%s%s", first ? "" : ";",
		    show_keyword_name ? "\"" : "", val,
		    show_keyword_name ? "\"" : "");
	    val = strchr (val, '\0') + 1;
	    first = 0;
	  }
	putchar ('\n');
      }
      break;
    case byte:
      {
	const char *val = nl_langinfo (item->item_id);

	if (show_keyword_name)
	  printf ("%s=", item->name);

	if (val != NULL)
	  printf ("%d", *val == '\377' ? -1 : *val);
	putchar ('\n');
      }
      break;
    case bytearray:
      {
	const char *val = nl_langinfo (item->item_id);
	int cnt = val ? strlen (val) : 0;

	if (show_keyword_name)
	  printf ("%s=", item->name);

	while (cnt > 1)
	  {
	    printf ("%d;", *val == '\177' ? -1 : *val);
	    --cnt;
	    ++val;
	  }

	printf ("%d\n", cnt == 0 || *val == '\177' ? -1 : *val);
      }
      break;
    case word:
      {
	union { unsigned int word; char *string; } val;
	val.string = nl_langinfo (item->item_id);
	if (show_keyword_name)
	  printf ("%s=", item->name);

	printf ("%d\n", val.word);
      }
      break;
    case wordarray:
      {
	int first = 1;
	union { unsigned int *wordarray; char *string; } val;

	val.string = nl_langinfo (item->item_id);
	if (show_keyword_name)
	  printf ("%s=", item->name);

	for (int cnt = 0; cnt < item->max; ++cnt)
	  {
	    printf ("%s%d", first ? "" : ";", val.wordarray[cnt]);
	    first = 0;
	  }
	putchar ('\n');
      }
      break;
    case wstring:
    case wstringarray:
    case wstringlist:
      /* We don't print wide character information since the same
	 information is available in a multibyte string.  */
    default:
      break;
    }
}

/* Show the information request for NAME.  */
static void
show_info (const char *name)
{
  for (size_t cat_no = 0; cat_no < NCATEGORIES; ++cat_no)
    if (cat_no != LC_ALL)
      {
	if (strcmp (name, category[cat_no].name) == 0)
	  /* Print the whole category.  */
	  {
	    if (show_category_name != 0)
	      puts (category[cat_no].name);

	    for (size_t item_no = 0;
		 item_no < category[cat_no].number;
		 ++item_no)
	      print_item (&category[cat_no].item_desc[item_no]);

	    return;
	  }

	for (size_t item_no = 0; item_no < category[cat_no].number; ++item_no)
	  if (strcmp (name, category[cat_no].item_desc[item_no].name) == 0)
	    {
	      if (show_category_name != 0)
		puts (category[cat_no].name);

	      print_item (&category[cat_no].item_desc[item_no]);
	      return;
	    }
      }

  /* The name is not a standard one.
     For testing and perhaps advanced use allow some more symbols.  */
  locale_special (name, show_category_name, show_keyword_name);
}

/* Set to true by try_setlocale if setlocale fails.  Used by
   setlocale_diagnostics.  */
static bool setlocale_failed;

/* Call setlocale, with non-fatal error reporting.  */
static void
try_setlocale (int category, const char *category_name)
{
  if (setlocale (category, "") == NULL)
    {
      error (0, errno, gettext ("Cannot set %s to default locale"),
	     category_name);
      setlocale_failed = true;
    }
}

/* Return a quoted version of the passed string, or NULL on error.  */
static char *
quote_string (const char *input)
{
  char *buffer;
  size_t length;
  FILE *stream = open_memstream (&buffer, &length);
  if (stream == NULL)
    return NULL;

  while (true)
    {
      unsigned char ch = *input++;
      if (ch == '\0')
	break;

      /* Use C backslash escapes for those control characters for
         which they are defined.  */
      switch (ch)
        {
          case '\a':
            putc_unlocked ('\\', stream);
            putc_unlocked ('a', stream);
            break;
          case '\b':
            putc_unlocked ('\\', stream);
            putc_unlocked ('b', stream);
            break;
          case '\f':
            putc_unlocked ('\\', stream);
            putc_unlocked ('f', stream);
            break;
          case '\n':
            putc_unlocked ('\\', stream);
            putc_unlocked ('n', stream);
            break;
          case '\r':
            putc_unlocked ('\\', stream);
            putc_unlocked ('r', stream);
            break;
          case '\t':
            putc_unlocked ('\\', stream);
            putc_unlocked ('t', stream);
            break;
          case '\v':
            putc_unlocked ('\\', stream);
            putc_unlocked ('v', stream);
            break;
          case '\\':
          case '\'':
          case '\"':
            putc_unlocked ('\\', stream);
            putc_unlocked (ch, stream);
            break;
        default:
          if (ch < ' ' || ch > '~')
            /* Use octal sequences because they are fixed width,
               unlike hexadecimal sequences.  */
            fprintf (stream, "\\%03o", ch);
          else
            putc_unlocked (ch, stream);
        }
    }

  if (ferror (stream))
    {
      fclose (stream);
      free (buffer);
      return NULL;
    }
  if (fclose (stream) != 0)
    {
      free (buffer);
      return NULL;
    }

  return buffer;
}

/* Print additional information if there was a setlocale error (during
   try_setlocale).  */
static void
setlocale_diagnostics (void)
{
  if (setlocale_failed)
    {
      const char *locpath = getenv ("LOCPATH");
      if (locpath != NULL)
	{
	  char *quoted = quote_string (locpath);
	  if (quoted != NULL)
	    fprintf (stderr,
		     gettext ("\
warning: The LOCPATH variable is set to \"%s\"\n"),
		     quoted);
	  else
	    fputs ("warning: The LOCPATH variable is set\n", stderr);
	  free (quoted);
	}
    }
}
