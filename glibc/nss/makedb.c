/* Create simple DB database from textual input.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1996.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <argp.h>
#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libintl.h>
#include <locale.h>
#include <search.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/uio.h>
#include "nss_db/nss_db.h"
#include <libc-diag.h>

/* Get libc version number.  */
#include "../version.h"

/* The hashing function we use.  */
#include "../intl/hash-string.h"

/* SELinux support.  */
#ifdef HAVE_SELINUX
# include <selinux/selinux.h>
#endif

#ifndef MAP_POPULATE
# define MAP_POPULATE 0
#endif

#define PACKAGE _libc_intl_domainname

/* List of data bases.  */
struct database
{
  char dbid;
  bool extra_string;
  struct database *next;
  void *entries;
  size_t nentries;
  size_t nhashentries;
  stridx_t *hashtable;
  size_t keystrlen;
  stridx_t *keyidxtab;
  char *keystrtab;
} *databases;
static size_t ndatabases;
static size_t nhashentries_total;
static size_t valstrlen;
static void *valstrtree;
static char *valstrtab;
static size_t extrastrlen;

/* Database entry.  */
struct dbentry
{
  stridx_t validx;
  uint32_t hashval;
  char str[0];
};

/* Stored string entry.  */
struct valstrentry
{
  stridx_t idx;
  bool extra_string;
  char str[0];
};


/* True if any entry has been added.  */
static bool any_dbentry;

/* If non-zero convert key to lower case.  */
static int to_lowercase;

/* If non-zero print content of input file, one entry per line.  */
static int do_undo;

/* If non-zero do not print informational messages.  */
static int be_quiet;

/* Name of output file.  */
static const char *output_name;

/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { "fold-case", 'f', NULL, 0, N_("Convert key to lower case") },
  { "output", 'o', N_("NAME"), 0, N_("Write output to file NAME") },
  { "quiet", 'q', NULL, 0,
    N_("Do not print messages while building database") },
  { "undo", 'u', NULL, 0,
    N_("Print content of database file, one entry a line") },
  { "generated", 'g', N_("CHAR"), 0,
    N_("Generated line not part of iteration") },
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("Create simple database from textual input.");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("\
INPUT-FILE OUTPUT-FILE\n-o OUTPUT-FILE INPUT-FILE\n-u INPUT-FILE");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help
};


/* List of databases which are not part of the iteration table.  */
static struct db_option
{
  char dbid;
  struct db_option *next;
} *db_options;


/* Prototypes for local functions.  */
static int process_input (FILE *input, const char *inname,
			  int to_lowercase, int be_quiet);
static int print_database (int fd);
static void compute_tables (void);
static int write_output (int fd);

/* SELinux support.  */
#ifdef HAVE_SELINUX
/* Set the SELinux file creation context for the given file. */
static void set_file_creation_context (const char *outname, mode_t mode);
static void reset_file_creation_context (void);
#else
# define set_file_creation_context(_outname,_mode)
# define reset_file_creation_context()
#endif


/* External functions.  */
#include <programs/xmalloc.h>


int
main (int argc, char *argv[])
{
  const char *input_name;
  FILE *input_file;
  int remaining;
  int mode = 0644;

  /* Set locale via LC_ALL.  */
  setlocale (LC_ALL, "");

  /* Set the text message domain.  */
  textdomain (_libc_intl_domainname);

  /* Initialize local variables.  */
  input_name = NULL;

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  /* Determine file names.  */
  if (do_undo || output_name != NULL)
    {
      if (remaining + 1 != argc)
	{
	wrong_arguments:
	  error (0, 0, gettext ("wrong number of arguments"));
	  argp_help (&argp, stdout, ARGP_HELP_SEE,
		     program_invocation_short_name);
	  exit (1);
	}
      input_name = argv[remaining];
    }
  else
    {
      if (remaining + 2 != argc)
	goto wrong_arguments;

      input_name = argv[remaining++];
      output_name = argv[remaining];
    }

  /* Special handling if we are asked to print the database.  */
  if (do_undo)
    {
      int fd = open (input_name, O_RDONLY);
      if (fd == -1)
	error (EXIT_FAILURE, errno, gettext ("cannot open database file `%s'"),
	       input_name);

      int status = print_database (fd);

      close (fd);

      return status;
    }

  /* Open input file.  */
  if (strcmp (input_name, "-") == 0 || strcmp (input_name, "/dev/stdin") == 0)
    input_file = stdin;
  else
    {
      struct stat64 st;

      input_file = fopen64 (input_name, "r");
      if (input_file == NULL)
	error (EXIT_FAILURE, errno, gettext ("cannot open input file `%s'"),
	       input_name);

      /* Get the access rights from the source file.  The output file should
	 have the same.  */
      if (fstat64 (fileno (input_file), &st) >= 0)
	mode = st.st_mode & ACCESSPERMS;
    }

  /* Start the real work.  */
  int status = process_input (input_file, input_name, to_lowercase, be_quiet);

  /* Close files.  */
  if (input_file != stdin)
    fclose (input_file);

  /* No need to continue when we did not read the file successfully.  */
  if (status != EXIT_SUCCESS)
    return status;

  /* Bail out if nothing is to be done.  */
  if (!any_dbentry)
    {
      if (be_quiet)
	return EXIT_SUCCESS;
      else
	error (EXIT_SUCCESS, 0, gettext ("no entries to be processed"));
    }

  /* Compute hash and string tables.  */
  compute_tables ();

  /* Open output file.  This must not be standard output so we don't
     handle "-" and "/dev/stdout" special.  */
  char *tmp_output_name;
  if (asprintf (&tmp_output_name, "%s.XXXXXX", output_name) == -1)
    error (EXIT_FAILURE, errno, gettext ("cannot create temporary file name"));

  set_file_creation_context (output_name, mode);
  int fd = mkstemp (tmp_output_name);
  reset_file_creation_context ();
  if (fd == -1)
    error (EXIT_FAILURE, errno, gettext ("cannot create temporary file"));

  status = write_output (fd);

  if (status == EXIT_SUCCESS)
    {
      struct stat64 st;

      if (fstat64 (fd, &st) == 0)
	{
	  if ((st.st_mode & ACCESSPERMS) != mode)
	    /* We ignore problems with changing the mode.  */
	    fchmod (fd, mode);
	}
      else
	{
	  error (0, errno, gettext ("cannot stat newly created file"));
	  status = EXIT_FAILURE;
	}
    }

  close (fd);

  if (status == EXIT_SUCCESS)
    {
      if (rename (tmp_output_name, output_name) != 0)
	{
	  error (0, errno, gettext ("cannot rename temporary file"));
	  status = EXIT_FAILURE;
	  goto do_unlink;
	}
    }
  else
  do_unlink:
    unlink (tmp_output_name);

  return status;
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  struct db_option *newp;

  switch (key)
    {
    case 'f':
      to_lowercase = 1;
      break;
    case 'o':
      output_name = arg;
      break;
    case 'q':
      be_quiet = 1;
      break;
    case 'u':
      do_undo = 1;
      break;
    case 'g':
      newp = xmalloc (sizeof (*newp));
      newp->dbid = arg[0];
      newp->next = db_options;
      db_options = newp;
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
  fprintf (stream, "makedb %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


static int
dbentry_compare (const void *p1, const void *p2)
{
  const struct dbentry *d1 = (const struct dbentry *) p1;
  const struct dbentry *d2 = (const struct dbentry *) p2;

  if (d1->hashval != d2->hashval)
    return d1->hashval < d2->hashval ? -1 : 1;

  return strcmp (d1->str, d2->str);
}


static int
valstr_compare (const void *p1, const void *p2)
{
  const struct valstrentry *d1 = (const struct valstrentry *) p1;
  const struct valstrentry *d2 = (const struct valstrentry *) p2;

  return strcmp (d1->str, d2->str);
}


static int
process_input (FILE *input, const char *inname, int to_lowercase, int be_quiet)
{
  char *line;
  size_t linelen;
  int status;
  size_t linenr;

  line = NULL;
  linelen = 0;
  status = EXIT_SUCCESS;
  linenr = 0;

  struct database *last_database = NULL;

  while (!feof_unlocked (input))
    {
      ssize_t n = getline (&line, &linelen, input);
      if (n < 0)
	/* This means end of file or some bug.  */
	break;
      if (n == 0)
	/* Short read.  Probably interrupted system call. */
	continue;

      ++linenr;

      if (line[n - 1] == '\n')
	/* Remove trailing newline.  */
	line[--n] = '\0';

      char *cp = line;
      while (isspace (*cp))
	++cp;

      if (*cp == '#' || *cp == '\0')
	/* First non-space character in line '#': it's a comment.
	   Also go to the next line if it is empty except for whitespaces. */
	continue;

      /* Skip over the character indicating the database so that it is not
	 affected by TO_LOWERCASE.  */
      char *key = cp++;
      while (*cp != '\0' && !isspace (*cp))
	{
	  if (to_lowercase)
	    *cp = tolower (*cp);
	  ++cp;
	}

      if (*cp == '\0')
	/* It's a line without a value field.  */
	continue;

      *cp++ = '\0';
      size_t keylen = cp - key;

      while (isspace (*cp))
	++cp;

      char *data = cp;
      size_t datalen = (&line[n] - cp) + 1;

      /* Find the database.  */
      if (last_database == NULL || last_database->dbid != key[0])
	{
	  last_database = databases;
	  while (last_database != NULL && last_database->dbid != key[0])
	    last_database = last_database->next;

	  if (last_database == NULL)
	    {
	      last_database = xmalloc (sizeof (*last_database));
	      last_database->dbid = key[0];
	      last_database->extra_string = false;
	      last_database->next = databases;
	      last_database->entries = NULL;
	      last_database->nentries = 0;
	      last_database->keystrlen = 0;
	      databases = last_database;

	      struct db_option *runp = db_options;
	      while (runp != NULL)
		if (runp->dbid == key[0])
		  {
		    last_database->extra_string = true;
		    break;
		  }
		else
		  runp = runp->next;
	    }
	}

      /* Skip the database selector.  */
      ++key;
      --keylen;

      /* Store the data.  */
      struct valstrentry *nentry = xmalloc (sizeof (struct valstrentry)
					    + datalen);
      if (last_database->extra_string)
	nentry->idx = extrastrlen;
      else
	nentry->idx = valstrlen;
      nentry->extra_string = last_database->extra_string;
      memcpy (nentry->str, data, datalen);

      struct valstrentry **fdata = tsearch (nentry, &valstrtree,
					    valstr_compare);
      if (fdata == NULL)
	error (EXIT_FAILURE, errno, gettext ("cannot create search tree"));

      if (*fdata != nentry)
	{
	  /* We can reuse a string.  */
	  free (nentry);
	  nentry = *fdata;
	}
      else
	if (last_database->extra_string)
	  extrastrlen += datalen;
	else
	  valstrlen += datalen;

      /* Store the key.  */
      struct dbentry *newp = xmalloc (sizeof (struct dbentry) + keylen);
      newp->validx = nentry->idx;
      newp->hashval = __hash_string (key);
      memcpy (newp->str, key, keylen);

      struct dbentry **found = tsearch (newp, &last_database->entries,
					dbentry_compare);
      if (found == NULL)
	error (EXIT_FAILURE, errno, gettext ("cannot create search tree"));

      if (*found != newp)
	{
	  free (newp);
	  if (!be_quiet)
	    error_at_line (0, 0, inname, linenr, gettext ("duplicate key"));
	  continue;
	}

      ++last_database->nentries;
      last_database->keystrlen += keylen;

      any_dbentry = true;
    }

  if (ferror_unlocked (input))
    {
      error (0, 0, gettext ("problems while reading `%s'"), inname);
      status = EXIT_FAILURE;
    }

  return status;
}


static void
copy_valstr (const void *nodep, const VISIT which, const int depth)
{
  if (which != leaf && which != postorder)
    return;

  const struct valstrentry *p = *(const struct valstrentry **) nodep;

  strcpy (valstrtab + (p->extra_string ? valstrlen : 0) + p->idx, p->str);
}


/* Determine if the candidate is prime by using a modified trial division
   algorithm. The candidate must be both odd and greater than 4.  */
static int
is_prime (size_t candidate)
{
  size_t divn = 3;
  size_t sq = divn * divn;

  assert (candidate > 4 && candidate % 2 != 0);

  while (sq < candidate && candidate % divn != 0)
    {
      ++divn;
      sq += 4 * divn;
      ++divn;
    }

  return candidate % divn != 0;
}


static size_t
next_prime (size_t seed)
{
  /* Make sure that we're always greater than 4.  */
  seed = (seed + 4) | 1;

  while (!is_prime (seed))
    seed += 2;

  return seed;
}

#ifndef NESTING
static struct database *globdb;
static size_t max_chainlength;
static char *wp;
static size_t nhashentries;
static bool copy_string;

void add_key(const void *nodep, const VISIT which, const int depth)
{
  if (which != leaf && which != postorder)
    return;

  const struct dbentry *dbe = *(const struct dbentry **) nodep;

  ptrdiff_t stridx;
  if (copy_string)
    {
      stridx = wp - globdb->keystrtab;
      wp = stpcpy (wp, dbe->str) + 1;
    }
  else
    stridx = 0;

  size_t hidx = dbe->hashval % nhashentries;
  size_t hval2 = 1 + dbe->hashval % (nhashentries - 2);
  size_t chainlength = 0;

  while (globdb->hashtable[hidx] != ~((stridx_t) 0))
    {
      ++chainlength;
      if ((hidx += hval2) >= nhashentries)
	hidx -= nhashentries;
    }

  globdb->hashtable[hidx] = ((globdb->extra_string ? valstrlen : 0)
			 + dbe->validx);
  globdb->keyidxtab[hidx] = stridx;

  max_chainlength = MAX (max_chainlength, chainlength);
}
#endif

static void
compute_tables (void)
{
  valstrtab = xmalloc (roundup (valstrlen + extrastrlen, sizeof (stridx_t)));
  while ((valstrlen + extrastrlen) % sizeof (stridx_t) != 0)
    valstrtab[valstrlen++] = '\0';
  twalk (valstrtree, copy_valstr);

  static struct database *db;
  for (db = databases; db != NULL; db = db->next)
    if (db->nentries != 0)
      {
	++ndatabases;

	/* We simply use an odd number large than twice the number of
	   elements to store in the hash table for the size.  This gives
	   enough efficiency.  */
#define TEST_RANGE 30
	size_t nhashentries_min = next_prime (db->nentries < TEST_RANGE
					      ? db->nentries
					      : db->nentries * 2 - TEST_RANGE);
	size_t nhashentries_max = MAX (nhashentries_min, db->nentries * 4);
	size_t nhashentries_best = nhashentries_min;
	size_t chainlength_best = db->nentries;

	db->hashtable = xmalloc (2 * nhashentries_max * sizeof (stridx_t)
				 + db->keystrlen);
	db->keyidxtab = db->hashtable + nhashentries_max;
	db->keystrtab = (char *) (db->keyidxtab + nhashentries_max);

#ifdef NESTING
	static size_t max_chainlength;
	static char *wp;
	static size_t nhashentries;
	static bool copy_string;

	void add_key(const void *nodep, const VISIT which, const int depth)
	{
	  if (which != leaf && which != postorder)
	    return;

	  const struct dbentry *dbe = *(const struct dbentry **) nodep;

	  ptrdiff_t stridx;
	  if (copy_string)
	    {
	      stridx = wp - db->keystrtab;
	      wp = stpcpy (wp, dbe->str) + 1;
	    }
	  else
	    stridx = 0;

	  size_t hidx = dbe->hashval % nhashentries;
	  size_t hval2 = 1 + dbe->hashval % (nhashentries - 2);
	  size_t chainlength = 0;

	  while (db->hashtable[hidx] != ~((stridx_t) 0))
	    {
	      ++chainlength;
	      if ((hidx += hval2) >= nhashentries)
		hidx -= nhashentries;
	    }

	  db->hashtable[hidx] = ((db->extra_string ? valstrlen : 0)
				 + dbe->validx);
	  db->keyidxtab[hidx] = stridx;

	  max_chainlength = MAX (max_chainlength, chainlength);
	}
#else
	globdb = db;
#endif

	copy_string = false;
	nhashentries = nhashentries_min;
	for (size_t cnt = 0; cnt < TEST_RANGE; ++cnt)
	  {
	    memset (db->hashtable, '\xff', nhashentries * sizeof (stridx_t));

	    max_chainlength = 0;
	    wp = db->keystrtab;

	    twalk (db->entries, add_key);

	    if (max_chainlength == 0)
	      {
		/* No need to look further, this is as good as it gets.  */
		nhashentries_best = nhashentries;
		break;
	      }

	    if (max_chainlength < chainlength_best)
	      {
		chainlength_best = max_chainlength;
		nhashentries_best = nhashentries;
	      }

	    nhashentries = next_prime (nhashentries + 1);
	    if (nhashentries > nhashentries_max)
	      break;
	  }

	/* Recompute the best table again, this time fill in the strings.  */
	nhashentries = nhashentries_best;
	memset (db->hashtable, '\xff',
		2 * nhashentries_max * sizeof (stridx_t));
	copy_string = true;
	wp = db->keystrtab;

	twalk (db->entries, add_key);

	db->nhashentries = nhashentries_best;
	nhashentries_total += nhashentries_best;
    }
}


static int
write_output (int fd)
{
  struct nss_db_header *header;
  uint64_t file_offset = (sizeof (struct nss_db_header)
			  + (ndatabases * sizeof (header->dbs[0])));
  header = alloca (file_offset);

  header->magic = NSS_DB_MAGIC;
  header->ndbs = ndatabases;
  header->valstroffset = file_offset;
  header->valstrlen = valstrlen;

  size_t filled_dbs = 0;
  size_t iov_nelts = 2 + ndatabases * 3;
  struct iovec iov[iov_nelts];
  iov[0].iov_base = header;
  iov[0].iov_len = file_offset;

  iov[1].iov_base = valstrtab;
  iov[1].iov_len = valstrlen + extrastrlen;
  file_offset += iov[1].iov_len;

  size_t keydataoffset = file_offset + nhashentries_total * sizeof (stridx_t);
  for (struct database *db = databases; db != NULL; db = db->next)
    if (db->entries != NULL)
      {
	assert (file_offset % sizeof (stridx_t) == 0);
	assert (filled_dbs < ndatabases);

	header->dbs[filled_dbs].id = db->dbid;
	memset (header->dbs[filled_dbs].pad, '\0',
		sizeof (header->dbs[0].pad));
	header->dbs[filled_dbs].hashsize = db->nhashentries;

	iov[2 + filled_dbs].iov_base = db->hashtable;
	iov[2 + filled_dbs].iov_len = db->nhashentries * sizeof (stridx_t);
	header->dbs[filled_dbs].hashoffset = file_offset;
	file_offset += iov[2 + filled_dbs].iov_len;

	iov[2 + ndatabases + filled_dbs * 2].iov_base = db->keyidxtab;
	iov[2 + ndatabases + filled_dbs * 2].iov_len
	  = db->nhashentries * sizeof (stridx_t);
	header->dbs[filled_dbs].keyidxoffset = keydataoffset;
	keydataoffset += iov[2 + ndatabases + filled_dbs * 2].iov_len;

	iov[3 + ndatabases + filled_dbs * 2].iov_base = db->keystrtab;
	iov[3 + ndatabases + filled_dbs * 2].iov_len = db->keystrlen;
	header->dbs[filled_dbs].keystroffset = keydataoffset;
	keydataoffset += iov[3 + ndatabases + filled_dbs * 2].iov_len;

	++filled_dbs;
      }

  assert (filled_dbs == ndatabases);
  assert (file_offset == (iov[0].iov_len + iov[1].iov_len
			  + nhashentries_total * sizeof (stridx_t)));
  header->allocate = file_offset;

#if __GNUC_PREREQ (10, 0) && !__GNUC_PREREQ (11, 0)
  DIAG_PUSH_NEEDS_COMMENT;
  /* Avoid GCC 10 false positive warning: specified size exceeds maximum
     object size.  */
  DIAG_IGNORE_NEEDS_COMMENT (10, "-Wstringop-overflow");
#endif

  assert (iov_nelts <= INT_MAX);
  if (writev (fd, iov, iov_nelts) != keydataoffset)
    {
      error (0, errno, gettext ("failed to write new database file"));
      return EXIT_FAILURE;
    }

#if __GNUC_PREREQ (10, 0) && !__GNUC_PREREQ (11, 0)
  DIAG_POP_NEEDS_COMMENT;
#endif

  return EXIT_SUCCESS;
}


static int
print_database (int fd)
{
  struct stat64 st;
  if (fstat64 (fd, &st) != 0)
    error (EXIT_FAILURE, errno, gettext ("cannot stat database file"));

  const struct nss_db_header *header = mmap (NULL, st.st_size, PROT_READ,
					     MAP_PRIVATE|MAP_POPULATE, fd, 0);
  if (header == MAP_FAILED)
    error (EXIT_FAILURE, errno, gettext ("cannot map database file"));

  if (header->magic != NSS_DB_MAGIC)
    error (EXIT_FAILURE, 0, gettext ("file not a database file"));

  const char *valstrtab = (const char *) header + header->valstroffset;

  for (unsigned int dbidx = 0; dbidx < header->ndbs; ++dbidx)
    {
      const stridx_t *stridxtab
	= ((const stridx_t *) ((const char *) header
			       + header->dbs[dbidx].keyidxoffset));
      const char *keystrtab
	= (const char *) header + header->dbs[dbidx].keystroffset;
      const stridx_t *hashtab
	= (const stridx_t *) ((const char *) header
			      + header->dbs[dbidx].hashoffset);

      for (uint32_t hidx = 0; hidx < header->dbs[dbidx].hashsize; ++hidx)
	if (hashtab[hidx] != ~((stridx_t) 0))
	  printf ("%c%s %s\n",
		  header->dbs[dbidx].id,
		  keystrtab + stridxtab[hidx],
		  valstrtab + hashtab[hidx]);
    }

  return EXIT_SUCCESS;
}


#ifdef HAVE_SELINUX

/* security_context_t and matchpathcon (along with several other symbols) were
   marked as deprecated by the SELinux API starting from version 3.1.  We use
   them here, but should eventually switch to the newer API.  */
DIAG_PUSH_NEEDS_COMMENT
DIAG_IGNORE_NEEDS_COMMENT (10, "-Wdeprecated-declarations");

static void
set_file_creation_context (const char *outname, mode_t mode)
{
  static int enabled;
  static int enforcing;
  security_context_t ctx;

  /* Check if SELinux is enabled, and remember. */
  if (enabled == 0)
    enabled = is_selinux_enabled () ? 1 : -1;
  if (enabled < 0)
    return;

  /* Check if SELinux is enforcing, and remember. */
  if (enforcing == 0)
    enforcing = security_getenforce () ? 1 : -1;

  /* Determine the context which the file should have. */
  ctx = NULL;
  if (matchpathcon (outname, S_IFREG | mode, &ctx) == 0 && ctx != NULL)
    {
      if (setfscreatecon (ctx) != 0)
	error (enforcing > 0 ? EXIT_FAILURE : 0, 0,
	       gettext ("cannot set file creation context for `%s'"),
	       outname);

      freecon (ctx);
    }
}
DIAG_POP_NEEDS_COMMENT

static void
reset_file_creation_context (void)
{
  setfscreatecon (NULL);
}
#endif
