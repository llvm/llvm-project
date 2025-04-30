/* Generate fastloading iconv module configuration files.
   Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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
#include <error.h>
#include <errno.h>
#include <fcntl.h>
#include <libintl.h>
#include <locale.h>
#include <mcheck.h>
#include <search.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdio_ext.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/cdefs.h>
#include <sys/uio.h>

#include "iconvconfig.h"
#include <gconv_parseconfdir.h>

/* Get libc version number.  */
#include "../version.h"

#define PACKAGE _libc_intl_domainname


/* The hashing function we use.  */
#include "../intl/hash-string.h"


/* Types used.  */
struct module
{
  char *fromname;
  struct Strent *fromname_strent;
  char *filename;
  struct Strent *filename_strent;
  const char *directory;
  struct Strent *directory_strent;
  struct module *next;
  int cost;
  struct Strent *toname_strent;
  char toname[0];
};

struct alias
{
  char *fromname;
  struct Strent *froment;
  struct module *module;
  struct Strent *toent;
  char toname[0];
};

struct name
{
  const char *name;
  struct Strent *strent;
  int module_idx;
  uint32_t hashval;
};

struct name_info
{
  const char *canonical_name;
  struct Strent *canonical_strent;

  struct module *from_internal;
  struct module *to_internal;

  struct other_conv_list
  {
    int dest_idx;
    struct other_conv
    {
      gidx_t module_idx;
      struct module *module;
      struct other_conv *next;
    } other_conv;
    struct other_conv_list *next;
  } *other_conv_list;
};


/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

/* Short description of program.  */
static const char doc[] = N_("\
Create fastloading iconv module configuration file.");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("[DIR...]");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Definitions of arguments for argp functions.  */
#define OPT_PREFIX 300
#define OPT_NOSTDLIB 301
static const struct argp_option options[] =
{
  { "prefix", OPT_PREFIX, N_("PATH"), 0,
    N_("Prefix used for all file accesses") },
  { "output", 'o', N_("FILE"), 0, N_("\
Put output in FILE instead of installed location\
 (--prefix does not apply to FILE)") },
  { "nostdlib", OPT_NOSTDLIB, NULL, 0,
    N_("Do not search standard directories, only those on the command line") },
  { NULL, 0, NULL, 0, NULL }
};

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help
};


/* The function doing the actual work.  */
static int handle_dir (const char *dir);

/* Add all known builtin conversions and aliases.  */
static void add_builtins (void);

/* Create list of all aliases without circular aliases.  */
static void get_aliases (void);

/* Create list of all modules.  */
static void get_modules (void);

/* Get list of all the names and thereby indexing them.  */
static void generate_name_list (void);

/* Collect information about all the names.  */
static void generate_name_info (void);

/* Write the output file.  */
static int write_output (void);


/* Prefix to be used for all file accesses.  */
static const char *prefix = "";
/* Its length.  */
static size_t prefix_len;

/* Directory to place output file in.  */
static const char *output_file;
/* Its length.  */
static size_t output_file_len;

/* If true, omit the GCONV_PATH directories and require some arguments.  */
static bool nostdlib;

/* Search tree of the modules we know.  */
static void *modules;

/* Search tree of the aliases we know.  */
static void *aliases;

/* Search tree for name to index mapping.  */
static void *names;

/* Number of names we know about.  */
static int nnames;

/* List of all aliases.  */
static struct alias **alias_list;
static size_t nalias_list;
static size_t nalias_list_max;

/* List of all modules.  */
static struct module **module_list;
static size_t nmodule_list;
static size_t nmodule_list_max;

/* Names and information about them.  */
static struct name_info *name_info;
static size_t nname_info;

/* Number of translations not from or to INTERNAL.  */
static size_t nextra_modules;


/* Names and aliases for the builtin transformations.  */
static struct
{
  const char *from;
  const char *to;
} builtin_alias[] =
  {
#define BUILTIN_ALIAS(alias, real) \
    { .from = alias, .to = real },
#define BUILTIN_TRANSFORMATION(From, To, Cost, Name, Fct, BtowcFct, \
			       MinF, MaxF, MinT, MaxT)
#include <gconv_builtin.h>
  };
#undef BUILTIN_ALIAS
#undef BUILTIN_TRANSFORMATION
#define nbuiltin_alias (sizeof (builtin_alias) / sizeof (builtin_alias[0]))

static struct
{
  const char *from;
  const char *to;
  const char *module;
  int cost;
} builtin_trans[] =
  {
#define BUILTIN_ALIAS(alias, real)
#define BUILTIN_TRANSFORMATION(From, To, Cost, Name, Fct, BtowcFct, \
			       MinF, MaxF, MinT, MaxT) \
    { .from = From, .to = To, .module = Name, .cost = Cost },
#include <gconv_builtin.h>
  };
#undef BUILTIN_ALIAS
#undef BUILTIN_TRANSFORMATION
#define nbuiltin_trans (sizeof (builtin_trans) / sizeof (builtin_trans[0]))


/* Filename extension for the modules.  */
#ifndef MODULE_EXT
# define MODULE_EXT ".so"
#endif
static const char gconv_module_ext[] = MODULE_EXT;


#include <programs/xmalloc.h>
#include <programs/xasprintf.h>


/* C string table handling.  */
struct Strtab;
struct Strent;

/* Create new C string table object in memory.  */
extern struct Strtab *strtabinit (void);

/* Free resources allocated for C string table ST.  */
extern void strtabfree (struct Strtab *st);

/* Add string STR (length LEN is != 0) to C string table ST.  */
extern struct Strent *strtabadd (struct Strtab *st, const char *str,
				 size_t len);

/* Finalize string table ST and store size in *SIZE and return a pointer.  */
extern void *strtabfinalize (struct Strtab *st, size_t *size);

/* Get offset in string table for string associated with SE.  */
extern size_t strtaboffset (struct Strent *se);

/* String table we construct.  */
static struct Strtab *strtab;



int
main (int argc, char *argv[])
{
  int remaining;
  int status = 0;

  /* Enable memory use testing.  */
  /* mcheck_pedantic (NULL); */
  mtrace ();

  /* Set locale via LC_ALL.  */
  setlocale (LC_ALL, "");

  /* Set the text message domain.  */
  textdomain (_libc_intl_domainname);

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  if (nostdlib && remaining == argc)
    error (2, 0, _("Directory arguments required when using --nostdlib"));

  /* Initialize the string table.  */
  strtab = strtabinit ();

  /* Handle all directories mentioned.  */
  while (remaining < argc)
    status |= handle_dir (argv[remaining++]);

  if (! nostdlib)
    {
      /* In any case also handle the standard directory.  */
      char *path = strdupa (GCONV_PATH), *tp = strsep (&path, ":");
      while (tp != NULL)
	{
	  status |= handle_dir (tp);

	  tp = strsep (&path, ":");
	}
    }

  /* Add the builtin transformations and aliases without overwriting
     anything.  */
  add_builtins ();

  /* Store aliases in an array.  */
  get_aliases ();

  /* Get list of all modules.  */
  get_modules ();

  /* Generate list of all the names we know to handle in some way.  */
  generate_name_list ();

  /* Now we know all the names we will handle, collect information
     about them.  */
  generate_name_info ();

  /* Write the output file, but only if we haven't seen any error.  */
  if (status == 0)
    status = write_output ();
  else
    error (1, 0, _("no output file produced because warnings were issued"));

  return status;
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case OPT_PREFIX:
      prefix = arg;
      prefix_len = strlen (prefix);
      break;
    case 'o':
      output_file = arg;
      output_file_len = strlen (output_file);
      break;
    case OPT_NOSTDLIB:
      nostdlib = true;
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
  fprintf (stream, "iconvconfig %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


static int
alias_compare (const void *p1, const void *p2)
{
  const struct alias *a1 = (const struct alias *) p1;
  const struct alias *a2 = (const struct alias *) p2;

  return strcmp (a1->fromname, a2->fromname);
}


static void
new_alias (const char *fromname, size_t fromlen, const char *toname,
	   size_t tolen)
{
  struct alias *newp;
  void **inserted;

  newp = (struct alias *) xmalloc (sizeof (struct alias) + fromlen + tolen);

  newp->fromname = mempcpy (newp->toname, toname, tolen);
  memcpy (newp->fromname, fromname, fromlen);
  newp->module = NULL;

  inserted = (void **) tsearch (newp, &aliases, alias_compare);
  if (inserted == NULL)
    error (EXIT_FAILURE, errno, gettext ("while inserting in search tree"));
  if (*inserted != newp)
    /* Something went wrong, free this entry.  */
    free (newp);
  else
    {
      newp->froment = strtabadd (strtab, newp->fromname, fromlen);
      newp->toent = strtabadd (strtab, newp->toname, tolen);
    }
}


/* Add new alias.  */
static void
add_alias (char *rp)
{
  /* We now expect two more string.  The strings are normalized
     (converted to UPPER case) and strored in the alias database.  */
  char *from;
  char *to;
  char *wp;

  while (isspace (*rp))
    ++rp;
  from = wp = rp;
  while (*rp != '\0' && !isspace (*rp))
    *wp++ = toupper (*rp++);
  if (*rp == '\0')
    /* There is no `to' string on the line.  Ignore it.  */
    return;
  *wp++ = '\0';
  to = ++rp;
  while (isspace (*rp))
    ++rp;
  while (*rp != '\0' && !isspace (*rp))
    *wp++ = toupper (*rp++);
  if (to == wp)
    /* No `to' string, ignore the line.  */
    return;
  *wp++ = '\0';

  assert (strlen (from) + 1 == (size_t) (to - from));
  assert (strlen (to) + 1 == (size_t) (wp - to));

  new_alias (from, to - from, to, wp - to);
}


static void
append_alias (const void *nodep, VISIT value, int level)
{
  if (value != leaf && value != postorder)
    return;

  if (nalias_list_max == nalias_list)
    {
      nalias_list_max += 50;
      alias_list = (struct alias **) xrealloc (alias_list,
					       (nalias_list_max
						* sizeof (struct alias *)));
    }

  alias_list[nalias_list++] = *(struct alias **) nodep;
}


static void
get_aliases (void)
{
  twalk (aliases, append_alias);
}


static int
module_compare (const void *p1, const void *p2)
{
  const struct module *m1 = (const struct module *) p1;
  const struct module *m2 = (const struct module *) p2;
  int result;

  result = strcmp (m1->fromname, m2->fromname);
  if (result == 0)
    result = strcmp (m1->toname, m2->toname);

  return result;
}


/* Create new module record.  */
static void
new_module (const char *fromname, size_t fromlen, const char *toname,
	    size_t tolen, const char *dir_in,
	    const char *filename, size_t filelen, int cost, size_t need_ext)
{
  struct module *new_module;
  size_t dirlen = strlen (dir_in) + 1;
  const char *directory = xstrdup (dir_in);
  char *tmp;
  void **inserted;

  new_module = (struct module *) xmalloc (sizeof (struct module)
					  + fromlen + tolen + filelen
					  + need_ext);

  new_module->fromname = mempcpy (new_module->toname, toname, tolen);

  new_module->filename = mempcpy (new_module->fromname, fromname, fromlen);

  new_module->cost = cost;
  new_module->next = NULL;

  tmp = mempcpy (new_module->filename, filename, filelen);
  if (need_ext)
    {
      memcpy (tmp - 1, gconv_module_ext, need_ext + 1);
      filelen += need_ext;
    }
  new_module->directory = directory;

  /* Now insert the new module data structure in our search tree.  */
  inserted = (void **) tsearch (new_module, &modules, module_compare);
  if (inserted == NULL)
    error (EXIT_FAILURE, errno, "while inserting in search tree");
  if (*inserted != new_module)
    free (new_module);
  else
    {
      new_module->fromname_strent = strtabadd (strtab, new_module->fromname,
					       fromlen);
      new_module->toname_strent = strtabadd (strtab, new_module->toname,
					     tolen);
      new_module->filename_strent = strtabadd (strtab, new_module->filename,
					       filelen);
      new_module->directory_strent = strtabadd (strtab, directory, dirlen);
    }
}


/* Add new module.  */
static void
add_module (char *rp, const char *directory,
	    size_t dirlen __attribute__ ((__unused__)),
	    int modcount __attribute__ ((__unused__)))
{
  /* We expect now
     1. `from' name
     2. `to' name
     3. filename of the module
     4. an optional cost value
  */
  char *from;
  char *to;
  char *module;
  char *wp;
  int need_ext;
  int cost;

  while (isspace (*rp))
    ++rp;
  from = rp;
  while (*rp != '\0' && !isspace (*rp))
    {
      *rp = toupper (*rp);
      ++rp;
    }
  if (*rp == '\0')
    return;
  *rp++ = '\0';
  to = wp = rp;
  while (isspace (*rp))
    ++rp;
  while (*rp != '\0' && !isspace (*rp))
    *wp++ = toupper (*rp++);
  if (*rp == '\0')
    return;
  *wp++ = '\0';
  do
    ++rp;
  while (isspace (*rp));
  module = wp;
  while (*rp != '\0' && !isspace (*rp))
    *wp++ = *rp++;
  if (*rp == '\0')
    {
      /* There is no cost, use one by default.  */
      *wp++ = '\0';
      cost = 1;
    }
  else
    {
      /* There might be a cost value.  */
      char *endp;

      *wp++ = '\0';
      cost = strtol (rp, &endp, 10);
      if (rp == endp || cost < 1)
	/* No useful information.  */
	cost = 1;
    }

  if (module[0] == '\0')
    /* No module name given.  */
    return;

  /* See whether we must add the ending.  */
  need_ext = 0;
  if ((size_t) (wp - module) < sizeof (gconv_module_ext)
      || memcmp (wp - sizeof (gconv_module_ext), gconv_module_ext,
		 sizeof (gconv_module_ext)) != 0)
    /* We must add the module extension.  */
    need_ext = sizeof (gconv_module_ext) - 1;

  assert (strlen (from) + 1 == (size_t) (to - from));
  assert (strlen (to) + 1 == (size_t) (module - to));
  assert (strlen (module) + 1 == (size_t) (wp - module));

  new_module (from, to - from, to, module - to, directory, module, wp - module,
	      cost, need_ext);
}

/* Read config files and add the data for this directory to cache.  */
static int
handle_dir (const char *dir)
{
  size_t dirlen = strlen (dir);
  bool found = false;

  char *fulldir = xasprintf ("%s%s%s", dir[0] == '/' ? prefix : "",
			     dir, dir[dirlen - 1] != '/' ? "/" : "");

  found = gconv_parseconfdir (fulldir, strlen (fulldir));

  if (!found)
    {
      error (0, errno, "failed to open gconv configuration files in `%s'",
	     dir);
      error (0, 0,
	     "ensure that the directory contains either a valid "
	     "gconv-modules file or a gconv-modules.d directory with "
	     "configuration files with names ending in .conf.");
    }

  free (fulldir);

  return found ? 0 : 1;
}


static void
append_module (const void *nodep, VISIT value, int level)
{
  struct module *mo;

  if (value != leaf && value != postorder)
    return;

  mo = *(struct module **) nodep;

  if (nmodule_list > 0
      && strcmp (module_list[nmodule_list - 1]->fromname, mo->fromname) == 0)
    {
      /* Same name.  */
      mo->next = module_list[nmodule_list - 1];
      module_list[nmodule_list - 1] = mo;

      return;
    }

  if (nmodule_list_max == nmodule_list)
    {
      nmodule_list_max += 50;
      module_list = (struct module **) xrealloc (module_list,
						 (nmodule_list_max
						  * sizeof (struct module *)));
    }

  module_list[nmodule_list++] = mo;
}


static void
get_modules (void)
{
  twalk (modules, append_module);
}


static void
add_builtins (void)
{
  size_t cnt;

  /* Add all aliases.  */
  for (cnt = 0; cnt < nbuiltin_alias; ++cnt)
    new_alias (builtin_alias[cnt].from,
	       strlen (builtin_alias[cnt].from) + 1,
	       builtin_alias[cnt].to,
	       strlen (builtin_alias[cnt].to) + 1);

  /* add the builtin transformations.  */
  for (cnt = 0; cnt < nbuiltin_trans; ++cnt)
    new_module (builtin_trans[cnt].from,
		strlen (builtin_trans[cnt].from) + 1,
		builtin_trans[cnt].to,
		strlen (builtin_trans[cnt].to) + 1,
		"", builtin_trans[cnt].module,
		strlen (builtin_trans[cnt].module) + 1,
		builtin_trans[cnt].cost, 0);
}


static int
name_compare (const void *p1, const void *p2)
{
  const struct name *n1 = (const struct name *) p1;
  const struct name *n2 = (const struct name *) p2;

  return strcmp (n1->name, n2->name);
}


static struct name *
new_name (const char *str, struct Strent *strent)
{
  struct name *newp = (struct name *) xmalloc (sizeof (struct name));

  newp->name = str;
  newp->strent = strent;
  newp->module_idx = -1;
  newp->hashval = __hash_string (str);

  ++nnames;

  return newp;
}


static void
generate_name_list (void)
{
  size_t i;

  /* A name we always need.  */
  tsearch (new_name ("INTERNAL", strtabadd (strtab, "INTERNAL",
					    sizeof ("INTERNAL"))),
	   &names, name_compare);

  for (i = 0; i < nmodule_list; ++i)
    {
      struct module *runp;

      if (strcmp (module_list[i]->fromname, "INTERNAL") != 0)
	tsearch (new_name (module_list[i]->fromname,
			   module_list[i]->fromname_strent),
		 &names, name_compare);

      for (runp = module_list[i]; runp != NULL; runp = runp->next)
	if (strcmp (runp->toname, "INTERNAL") != 0)
	  tsearch (new_name (runp->toname, runp->toname_strent),
		   &names, name_compare);
    }
}


static int
name_to_module_idx (const char *name, int add)
{
  struct name **res;
  struct name fake_name = { .name = name };
  int idx;

  res = (struct name **) tfind (&fake_name, &names, name_compare);
  if (res == NULL)
    abort ();

  idx = (*res)->module_idx;
  if (idx == -1 && add)
    /* No module index assigned yet.  */
    idx = (*res)->module_idx = nname_info++;

  return idx;
}


static void
generate_name_info (void)
{
  size_t i;
  int idx;

  name_info = (struct name_info *) xcalloc (nmodule_list + 1,
					    sizeof (struct name_info));

  /* First add a special entry for the INTERNAL name.  This must have
     index zero.  */
  idx = name_to_module_idx ("INTERNAL", 1);
  name_info[0].canonical_name = "INTERNAL";
  name_info[0].canonical_strent = strtabadd (strtab, "INTERNAL",
					     sizeof ("INTERNAL"));
  assert (nname_info == 1);

  for (i = 0; i < nmodule_list; ++i)
    {
      struct module *runp;

      for (runp = module_list[i]; runp != NULL; runp = runp->next)
	if (strcmp (runp->fromname, "INTERNAL") == 0)
	  {
	    idx = name_to_module_idx (runp->toname, 1);
	    name_info[idx].from_internal = runp;
	    assert (name_info[idx].canonical_name == NULL
		    || strcmp (name_info[idx].canonical_name,
			       runp->toname) == 0);
	    name_info[idx].canonical_name = runp->toname;
	    name_info[idx].canonical_strent = runp->toname_strent;
	  }
	else if (strcmp (runp->toname, "INTERNAL") == 0)
	  {
	    idx = name_to_module_idx (runp->fromname, 1);
	    name_info[idx].to_internal = runp;
	    assert (name_info[idx].canonical_name == NULL
		    || strcmp (name_info[idx].canonical_name,
			       runp->fromname) == 0);
	    name_info[idx].canonical_name = runp->fromname;
	    name_info[idx].canonical_strent = runp->fromname_strent;
	  }
	else
	  {
	    /* This is a transformation not to or from the INTERNAL
	       encoding.  */
	    int from_idx = name_to_module_idx (runp->fromname, 1);
	    int to_idx = name_to_module_idx (runp->toname, 1);
	    struct other_conv_list *newp;

	    newp = (struct other_conv_list *)
	      xmalloc (sizeof (struct other_conv_list));
	    newp->other_conv.module_idx = to_idx;
	    newp->other_conv.module = runp;
	    newp->other_conv.next = NULL; /* XXX Allow multiple module sequence */
	    newp->dest_idx = to_idx;
	    newp->next = name_info[from_idx].other_conv_list;
	    name_info[from_idx].other_conv_list = newp;
	    assert (name_info[from_idx].canonical_name == NULL
		    || strcmp (name_info[from_idx].canonical_name,
			       runp->fromname) == 0);
	    name_info[from_idx].canonical_name = runp->fromname;
	    name_info[from_idx].canonical_strent = runp->fromname_strent;

	    ++nextra_modules;
	  }
    }

  /* Now add the module index information for all the aliases.  */
  for (i = 0; i < nalias_list; ++i)
    {
      struct name fake_name = { .name = alias_list[i]->toname };
      struct name **tonamep;

      tonamep = (struct name **) tfind (&fake_name, &names, name_compare);
      if (tonamep != NULL)
	{
	  struct name *newp = new_name (alias_list[i]->fromname,
					alias_list[i]->froment);
	  newp->module_idx = (*tonamep)->module_idx;
	  tsearch (newp, &names, name_compare);
	}
    }
}


static int
is_prime (unsigned long int candidate)
{
  /* No even number and none less than 10 will be passed here.  */
  unsigned long int divn = 3;
  unsigned long int sq = divn * divn;

  while (sq < candidate && candidate % divn != 0)
    {
      ++divn;
      sq += 4 * divn;
      ++divn;
    }

  return candidate % divn != 0;
}


static uint32_t
next_prime (uint32_t seed)
{
  /* Make it definitely odd.  */
  seed |= 1;

  while (!is_prime (seed))
    seed += 2;

  return seed;
}


/* Format of the output file.

   Offset   Length       Description
   0000     4            Magic header bytes
   0004     2            Offset of string table (stoff)
   0006     2            Offset of name hashing table (hoff)
   0008     2            Hashing table size (hsize)
   000A     2            Offset of module table (moff)
   000C     2            Offset of other conversion module table (ooff)

   stoff    ???          String table

   hoff     8*hsize      Array of tuples
			    string table offset
			    module index

   moff     ???          Array of tuples
			    canonical name offset
			    from-internal module dir name offset
			    from-internal module name off
			    to-internal module dir name offset
			    to-internal module name offset
			    offset into other conversion table

   ooff     ???          One or more of
			    number of steps/modules
			    one or more of tuple
			      canonical name offset for output
			      module dir name offset
			      module name offset
			 (following last entry with step count 0)
*/

static struct hash_entry *hash_table;
static size_t hash_size;

/* Function to insert the names.  */
static void name_insert (const void *nodep, VISIT value, int level)
{
  struct name *name;
  unsigned int idx;
  unsigned int hval2;

  if (value != leaf && value != postorder)
    return;

  name = *(struct name **) nodep;
  idx = name->hashval % hash_size;
  hval2 = 1 + name->hashval % (hash_size - 2);

  while (hash_table[idx].string_offset != 0)
    if ((idx += hval2) >= hash_size)
      idx -= hash_size;

  hash_table[idx].string_offset = strtaboffset (name->strent);

  assert (name->module_idx != -1);
  hash_table[idx].module_idx = name->module_idx;
}

static int
write_output (void)
{
  int fd;
  char *string_table;
  size_t string_table_size;
  struct gconvcache_header header;
  struct module_entry *module_table;
  char *extra_table;
  char *cur_extra_table;
  size_t n;
  int idx;
  struct iovec iov[6];
  static const gidx_t null_word;
  size_t total;
  char finalname[prefix_len + sizeof GCONV_MODULES_CACHE];
  char tmpfname[(output_file == NULL ? sizeof finalname : output_file_len + 1)
		+ strlen (".XXXXXX")];

  /* Open the output file.  */
  if (output_file == NULL)
    {
      assert (GCONV_MODULES_CACHE[0] == '/');
      strcpy (stpcpy (mempcpy (tmpfname, prefix, prefix_len),
		      GCONV_MODULES_CACHE),
	      ".XXXXXX");
      strcpy (mempcpy (finalname, prefix, prefix_len), GCONV_MODULES_CACHE);
    }
  else
    strcpy (mempcpy (tmpfname, output_file, output_file_len), ".XXXXXX");
  fd = mkstemp (tmpfname);
  if (fd == -1)
    return 1;

  /* Create the string table.  */
  string_table = strtabfinalize (strtab, &string_table_size);

  /* Create the hashing table.  We know how many strings we have.
     Creating a perfect hash table is not reasonable here.  Therefore
     we use open hashing and a table size which is the next prime 50%
     larger than the number of strings.  */
  hash_size = next_prime (nnames + (nnames >> 1));
  hash_table = (struct hash_entry *) xcalloc (hash_size,
					      sizeof (struct hash_entry));
  /* Fill the hash table.  */
  twalk (names, name_insert);

  /* Create the section for the module list.  */
  module_table = (struct module_entry *) xcalloc (sizeof (struct module_entry),
						  nname_info);

  /* Allocate memory for the non-INTERNAL conversions.  The allocated
     memory can be more than is actually needed.  */
  extra_table = (char *) xcalloc (sizeof (struct extra_entry)
				  + sizeof (gidx_t)
				  + sizeof (struct extra_entry_module),
				  nextra_modules);
  cur_extra_table = extra_table;

  /* Fill in the module information.  */
  for (n = 0; n < nname_info; ++n)
    {
      module_table[n].canonname_offset =
	strtaboffset (name_info[n].canonical_strent);

      if (name_info[n].from_internal == NULL)
	{
	  module_table[n].fromdir_offset = 0;
	  module_table[n].fromname_offset = 0;
	}
      else
	{
	  module_table[n].fromdir_offset =
	    strtaboffset (name_info[n].from_internal->directory_strent);
	  module_table[n].fromname_offset =
	    strtaboffset (name_info[n].from_internal->filename_strent);
	}

      if (name_info[n].to_internal == NULL)
	{
	  module_table[n].todir_offset = 0;
	  module_table[n].toname_offset = 0;
	}
      else
	{
	  module_table[n].todir_offset =
	    strtaboffset (name_info[n].to_internal->directory_strent);
	  module_table[n].toname_offset =
	    strtaboffset (name_info[n].to_internal->filename_strent);
	}

      if (name_info[n].other_conv_list != NULL)
	{
	  struct other_conv_list *other = name_info[n].other_conv_list;

	  /* Store the reference.  We add 1 to distinguish the entry
	     at offset zero from the case where no extra modules are
	     available.  The file reader has to account for the
	     offset.  */
	  module_table[n].extra_offset = 1 + cur_extra_table - extra_table;

	  do
	    {
	      struct other_conv *runp;
	      struct extra_entry *extra;

	      /* Allocate new entry.  */
	      extra = (struct extra_entry *) cur_extra_table;
	      cur_extra_table += sizeof (struct extra_entry);
	      extra->module_cnt = 0;

	      runp = &other->other_conv;
	      do
		{
		  cur_extra_table += sizeof (struct extra_entry_module);
		  extra->module[extra->module_cnt].outname_offset =
		    runp->next == NULL
		    ? other->dest_idx : runp->next->module_idx;
		  extra->module[extra->module_cnt].dir_offset =
		    strtaboffset (runp->module->directory_strent);
		  extra->module[extra->module_cnt].name_offset =
		    strtaboffset (runp->module->filename_strent);
		  ++extra->module_cnt;

		  runp = runp->next;
		}
	      while (runp != NULL);

	      other = other->next;
	    }
	  while (other != NULL);

	  /* Final module_cnt is zero.  */
	  *((gidx_t *) cur_extra_table) = 0;
	  cur_extra_table += sizeof (gidx_t);
	}
    }

  /* Clear padding.  */
  memset (&header, 0, sizeof (struct gconvcache_header));

  header.magic = GCONVCACHE_MAGIC;

  iov[0].iov_base = &header;
  iov[0].iov_len = sizeof (struct gconvcache_header);
  total = iov[0].iov_len;

  header.string_offset = total;
  iov[1].iov_base = string_table;
  iov[1].iov_len = string_table_size;
  total += iov[1].iov_len;

  idx = 2;
  if ((string_table_size & (sizeof (gidx_t) - 1)) != 0)
    {
      iov[2].iov_base = (void *) &null_word;
      iov[2].iov_len = (sizeof (gidx_t)
			- (string_table_size & (sizeof (gidx_t) - 1)));
      total += iov[2].iov_len;
      ++idx;
    }

  header.hash_offset = total;
  header.hash_size = hash_size;
  iov[idx].iov_base = hash_table;
  iov[idx].iov_len = hash_size * sizeof (struct hash_entry);
  total += iov[idx].iov_len;
  ++idx;

  header.module_offset = total;
  iov[idx].iov_base = module_table;
  iov[idx].iov_len = nname_info * sizeof (struct module_entry);
  total += iov[idx].iov_len;
  ++idx;

  assert ((size_t) (cur_extra_table - extra_table)
	  <= ((sizeof (struct extra_entry) + sizeof (gidx_t)
	       + sizeof (struct extra_entry_module))
	      * nextra_modules));
  header.otherconv_offset = total;
  iov[idx].iov_base = extra_table;
  iov[idx].iov_len = cur_extra_table - extra_table;
  total += iov[idx].iov_len;
  ++idx;

  if ((size_t) TEMP_FAILURE_RETRY (writev (fd, iov, idx)) != total
      /* The file was created with mode 0600.  Make it world-readable.  */
      || fchmod (fd, 0644) != 0
      /* Rename the file, possibly replacing an old one.  */
      || rename (tmpfname, output_file ?: finalname) != 0)
    {
      int save_errno = errno;
      close (fd);
      unlink (tmpfname);
      error (EXIT_FAILURE, save_errno,
	     gettext ("cannot generate output file"));
    }

  close (fd);

  return 0;
}
