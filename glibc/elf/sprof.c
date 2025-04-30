/* Read and display shared object profiling data.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <dlfcn.h>
#include <elf.h>
#include <error.h>
#include <fcntl.h>
#include <inttypes.h>
#include <libintl.h>
#include <locale.h>
#include <obstack.h>
#include <search.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <ldsodefs.h>
#include <sys/gmon.h>
#include <sys/gmon_out.h>
#include <sys/mman.h>
#include <sys/param.h>
#include <sys/stat.h>

/* Get libc version number.  */
#include "../version.h"

#define PACKAGE _libc_intl_domainname


#include <endian.h>
#if BYTE_ORDER == BIG_ENDIAN
# define byteorder ELFDATA2MSB
# define byteorder_name "big-endian"
#elif BYTE_ORDER == LITTLE_ENDIAN
# define byteorder ELFDATA2LSB
# define byteorder_name "little-endian"
#else
# error "Unknown BYTE_ORDER " BYTE_ORDER
# define byteorder ELFDATANONE
#endif

#ifndef PATH_MAX
# define PATH_MAX 1024
#endif


extern int __profile_frequency (void);

/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

#define OPT_TEST	1

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { NULL, 0, NULL, 0, N_("Output selection:") },
  { "call-pairs", 'c', NULL, 0,
    N_("print list of count paths and their number of use") },
  { "flat-profile", 'p', NULL, 0,
    N_("generate flat profile with counts and ticks") },
  { "graph", 'q', NULL, 0, N_("generate call graph") },

  { "test", OPT_TEST, NULL, OPTION_HIDDEN, NULL },
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("Read and display shared object profiling data.");
//For bug reporting instructions, please see:\n
//<https://www.gnu.org/software/libc/bugs.html>.\n");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("SHOBJ [PROFDATA]");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help
};


/* Operation modes.  */
static enum
{
  NONE = 0,
  FLAT_MODE = 1 << 0,
  CALL_GRAPH_MODE = 1 << 1,
  CALL_PAIRS = 1 << 2,

  DEFAULT_MODE = FLAT_MODE | CALL_GRAPH_MODE
} mode;

/* Nozero for testing.  */
static int do_test;

/* Strcuture describing calls.  */
struct here_fromstruct
{
  struct here_cg_arc_record volatile *here;
  uint16_t link;
};

/* We define a special type to address the elements of the arc table.
   This is basically the `gmon_cg_arc_record' format but it includes
   the room for the tag and it uses real types.  */
struct here_cg_arc_record
{
  uintptr_t from_pc;
  uintptr_t self_pc;
  uint32_t count;
} __attribute__ ((packed));


struct known_symbol;
struct arc_list
{
  size_t idx;
  uintmax_t count;

  struct arc_list *next;
};

static struct obstack ob_list;


struct known_symbol
{
  const char *name;
  uintptr_t addr;
  size_t size;
  bool weak;
  bool hidden;

  uintmax_t ticks;
  uintmax_t calls;

  struct arc_list *froms;
  struct arc_list *tos;
};


struct shobj
{
  const char *name;		/* User-provided name.  */

  struct link_map *map;
  const char *dynstrtab;	/* Dynamic string table of shared object.  */
  const char *soname;		/* Soname of shared object.  */

  uintptr_t lowpc;
  uintptr_t highpc;
  unsigned long int kcountsize;
  size_t expected_size;		/* Expected size of profiling file.  */
  size_t tossize;
  size_t fromssize;
  size_t fromlimit;
  unsigned int hashfraction;
  int s_scale;

  void *symbol_map;
  size_t symbol_mapsize;
  const ElfW(Sym) *symtab;
  size_t symtab_size;
  const char *strtab;

  struct obstack ob_str;
  struct obstack ob_sym;
};


struct real_gmon_hist_hdr
{
  char *low_pc;
  char *high_pc;
  int32_t hist_size;
  int32_t prof_rate;
  char dimen[15];
  char dimen_abbrev;
};


struct profdata
{
  void *addr;
  off_t size;

  char *hist;
  struct real_gmon_hist_hdr *hist_hdr;
  uint16_t *kcount;
  uint32_t narcs;		/* Number of arcs in toset.  */
  struct here_cg_arc_record *data;
  uint16_t *tos;
  struct here_fromstruct *froms;
};

/* Search tree for symbols.  */
static void *symroot;
static struct known_symbol **sortsym;
static size_t symidx;
static uintmax_t total_ticks;

/* Prototypes for local functions.  */
static struct shobj *load_shobj (const char *name);
static void unload_shobj (struct shobj *shobj);
static struct profdata *load_profdata (const char *name, struct shobj *shobj);
static void unload_profdata (struct profdata *profdata);
static void count_total_ticks (struct shobj *shobj, struct profdata *profdata);
static void count_calls (struct shobj *shobj, struct profdata *profdata);
static void read_symbols (struct shobj *shobj);
static void add_arcs (struct profdata *profdata);
static void generate_flat_profile (struct profdata *profdata);
static void generate_call_graph (struct profdata *profdata);
static void generate_call_pair_list (struct profdata *profdata);


int
main (int argc, char *argv[])
{
  const char *shobj;
  const char *profdata;
  struct shobj *shobj_handle;
  struct profdata *profdata_handle;
  int remaining;

  setlocale (LC_ALL, "");

  /* Initialize the message catalog.  */
  textdomain (_libc_intl_domainname);

  /* Parse and process arguments.  */
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  if (argc - remaining == 0 || argc - remaining > 2)
    {
      /* We need exactly two non-option parameter.  */
      argp_help (&argp, stdout, ARGP_HELP_SEE | ARGP_HELP_EXIT_ERR,
		 program_invocation_short_name);
      exit (1);
    }

  /* Get parameters.  */
  shobj = argv[remaining];
  if (argc - remaining == 2)
    profdata = argv[remaining + 1];
  else
    /* No filename for the profiling data given.  We will determine it
       from the soname of the shobj, later.  */
    profdata = NULL;

  /* First see whether we can load the shared object.  */
  shobj_handle = load_shobj (shobj);
  if (shobj_handle == NULL)
    exit (1);

  /* We can now determine the filename for the profiling data, if
     nececessary.  */
  if (profdata == NULL)
    {
      char *newp;
      const char *soname;
      size_t soname_len;

      soname = shobj_handle->soname ?: basename (shobj);
      soname_len = strlen (soname);
      newp = (char *) alloca (soname_len + sizeof ".profile");
      stpcpy (mempcpy (newp, soname, soname_len), ".profile");
      profdata = newp;
    }

  /* Now see whether the profiling data file matches the given object.   */
  profdata_handle = load_profdata (profdata, shobj_handle);
  if (profdata_handle == NULL)
    {
      unload_shobj (shobj_handle);

      exit (1);
    }

  read_symbols (shobj_handle);

  /* Count the ticks.  */
  count_total_ticks (shobj_handle, profdata_handle);

  /* Count the calls.  */
  count_calls (shobj_handle, profdata_handle);

  /* Add the arc information.  */
  add_arcs (profdata_handle);

  /* If no mode is specified fall back to the default mode.  */
  if (mode == NONE)
    mode = DEFAULT_MODE;

  /* Do some work.  */
  if (mode & FLAT_MODE)
    generate_flat_profile (profdata_handle);

  if (mode & CALL_GRAPH_MODE)
    generate_call_graph (profdata_handle);

  if (mode & CALL_PAIRS)
    generate_call_pair_list (profdata_handle);

  /* Free the resources.  */
  unload_shobj (shobj_handle);
  unload_profdata (profdata_handle);

  return 0;
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    case 'c':
      mode |= CALL_PAIRS;
      break;
    case 'p':
      mode |= FLAT_MODE;
      break;
    case 'q':
      mode |= CALL_GRAPH_MODE;
      break;
    case OPT_TEST:
      do_test = 1;
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
  fprintf (stream, "sprof %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"),
	   "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


/* Note that we must not use `dlopen' etc.  The shobj object must not
   be loaded for use.  */
static struct shobj *
load_shobj (const char *name)
{
  struct link_map *map = NULL;
  struct shobj *result;
  ElfW(Addr) mapstart = ~((ElfW(Addr)) 0);
  ElfW(Addr) mapend = 0;
  const ElfW(Phdr) *ph;
  size_t textsize;
  ElfW(Ehdr) *ehdr;
  int fd;
  ElfW(Shdr) *shdr;
  size_t pagesize = getpagesize ();

  /* Since we use dlopen() we must be prepared to work around the sometimes
     strange lookup rules for the shared objects.  If we have a file foo.so
     in the current directory and the user specfies foo.so on the command
     line (without specifying a directory) we should load the file in the
     current directory even if a normal dlopen() call would read the other
     file.  We do this by adding a directory portion to the name.  */
  if (strchr (name, '/') == NULL)
    {
      char *load_name = (char *) alloca (strlen (name) + 3);
      stpcpy (stpcpy (load_name, "./"), name);

      map = (struct link_map *) dlopen (load_name, RTLD_LAZY | __RTLD_SPROF);
    }
  if (map == NULL)
    {
      map = (struct link_map *) dlopen (name, RTLD_LAZY | __RTLD_SPROF);
      if (map == NULL)
	{
	  error (0, errno, _("failed to load shared object `%s'"), name);
	  return NULL;
	}
    }

  /* Prepare the result.  */
  result = (struct shobj *) calloc (1, sizeof (struct shobj));
  if (result == NULL)
    {
      error (0, errno, _("cannot create internal descriptor"));
      dlclose (map);
      return NULL;
    }
  result->name = name;
  result->map = map;

  /* Compute the size of the sections which contain program code.
     This must match the code in dl-profile.c (_dl_start_profile).  */
  for (ph = map->l_phdr; ph < &map->l_phdr[map->l_phnum]; ++ph)
    if (ph->p_type == PT_LOAD && (ph->p_flags & PF_X))
      {
	ElfW(Addr) start = (ph->p_vaddr & ~(pagesize - 1));
	ElfW(Addr) end = ((ph->p_vaddr + ph->p_memsz + pagesize - 1)
			  & ~(pagesize - 1));

	if (start < mapstart)
	  mapstart = start;
	if (end > mapend)
	  mapend = end;
      }

  result->lowpc = ROUNDDOWN ((uintptr_t) (mapstart + map->l_addr),
			     HISTFRACTION * sizeof (HISTCOUNTER));
  result->highpc = ROUNDUP ((uintptr_t) (mapend + map->l_addr),
			    HISTFRACTION * sizeof (HISTCOUNTER));
  if (do_test)
    printf ("load addr: %0#*" PRIxPTR "\n"
	    "lower bound PC: %0#*" PRIxPTR "\n"
	    "upper bound PC: %0#*" PRIxPTR "\n",
	    __ELF_NATIVE_CLASS == 32 ? 10 : 18, map->l_addr,
	    __ELF_NATIVE_CLASS == 32 ? 10 : 18, result->lowpc,
	    __ELF_NATIVE_CLASS == 32 ? 10 : 18, result->highpc);

  textsize = result->highpc - result->lowpc;
  result->kcountsize = textsize / HISTFRACTION;
  result->hashfraction = HASHFRACTION;
  if (do_test)
    printf ("hashfraction = %d\ndivider = %zu\n",
	    result->hashfraction,
	    result->hashfraction * sizeof (struct here_fromstruct));
  result->tossize = textsize / HASHFRACTION;
  result->fromlimit = textsize * ARCDENSITY / 100;
  if (result->fromlimit < MINARCS)
    result->fromlimit = MINARCS;
  if (result->fromlimit > MAXARCS)
    result->fromlimit = MAXARCS;
  result->fromssize = result->fromlimit * sizeof (struct here_fromstruct);

  result->expected_size = (sizeof (struct gmon_hdr)
			   + 4 + sizeof (struct gmon_hist_hdr)
			   + result->kcountsize
			   + 4 + 4
			   + (result->fromssize
			      * sizeof (struct here_cg_arc_record)));

  if (do_test)
    printf ("expected size: %zd\n", result->expected_size);

#define SCALE_1_TO_1	0x10000L

  if (result->kcountsize < result->highpc - result->lowpc)
    {
      size_t range = result->highpc - result->lowpc;
      size_t quot = range / result->kcountsize;

      if (quot >= SCALE_1_TO_1)
	result->s_scale = 1;
      else if (quot >= SCALE_1_TO_1 / 256)
	result->s_scale = SCALE_1_TO_1 / quot;
      else if (range > ULONG_MAX / 256)
	result->s_scale = ((SCALE_1_TO_1 * 256)
			   / (range / (result->kcountsize / 256)));
      else
	result->s_scale = ((SCALE_1_TO_1 * 256)
			   / ((range * 256) / result->kcountsize));
    }
  else
    result->s_scale = SCALE_1_TO_1;

  if (do_test)
    printf ("s_scale: %d\n", result->s_scale);

  /* Determine the dynamic string table.  */
  if (map->l_info[DT_STRTAB] == NULL)
    result->dynstrtab = NULL;
  else
    result->dynstrtab = (const char *) D_PTR (map, l_info[DT_STRTAB]);
  if (do_test)
    printf ("string table: %p\n", result->dynstrtab);

  /* Determine the soname.  */
  if (map->l_info[DT_SONAME] == NULL)
    result->soname = NULL;
  else
    result->soname = result->dynstrtab + map->l_info[DT_SONAME]->d_un.d_val;
  if (do_test && result->soname != NULL)
    printf ("soname: %s\n", result->soname);

  /* Now we have to load the symbol table.

     First load the section header table.  */
  ehdr = (ElfW(Ehdr) *) map->l_map_start;

  /* Make sure we are on the right party.  */
  if (ehdr->e_shentsize != sizeof (ElfW(Shdr)))
    abort ();

  /* And we need the shared object file descriptor again.  */
  fd = open (map->l_name, O_RDONLY);
  if (fd == -1)
    /* Dooh, this really shouldn't happen.  We know the file is available.  */
    error (EXIT_FAILURE, errno, _("Reopening shared object `%s' failed"),
	   map->l_name);

  /* Map the section header.  */
  size_t size = ehdr->e_shnum * sizeof (ElfW(Shdr));
  shdr = (ElfW(Shdr) *) alloca (size);
  if (pread (fd, shdr, size, ehdr->e_shoff) != size)
    error (EXIT_FAILURE, errno, _("reading of section headers failed"));

  /* Get the section header string table.  */
  char *shstrtab = (char *) alloca (shdr[ehdr->e_shstrndx].sh_size);
  if (pread (fd, shstrtab, shdr[ehdr->e_shstrndx].sh_size,
	     shdr[ehdr->e_shstrndx].sh_offset)
      != shdr[ehdr->e_shstrndx].sh_size)
    error (EXIT_FAILURE, errno,
	   _("reading of section header string table failed"));

  /* Search for the ".symtab" section.  */
  ElfW(Shdr) *symtab_entry = NULL;
  ElfW(Shdr) *debuglink_entry = NULL;
  for (int idx = 0; idx < ehdr->e_shnum; ++idx)
    if (shdr[idx].sh_type == SHT_SYMTAB
	&& strcmp (shstrtab + shdr[idx].sh_name, ".symtab") == 0)
      {
	symtab_entry = &shdr[idx];
	break;
      }
    else if (shdr[idx].sh_type == SHT_PROGBITS
	     && strcmp (shstrtab + shdr[idx].sh_name, ".gnu_debuglink") == 0)
      debuglink_entry = &shdr[idx];

  /* Get the file name of the debuginfo file if necessary.  */
  int symfd = fd;
  if (symtab_entry == NULL && debuglink_entry != NULL)
    {
      size_t size = debuglink_entry->sh_size;
      char *debuginfo_fname = (char *) alloca (size + 1);
      debuginfo_fname[size] = '\0';
      if (pread (fd, debuginfo_fname, size, debuglink_entry->sh_offset)
	  != size)
	{
	  fprintf (stderr, _("*** Cannot read debuginfo file name: %m\n"));
	  goto no_debuginfo;
	}

      static const char procpath[] = "/proc/self/fd/%d";
      char origprocname[sizeof (procpath) + sizeof (int) * 3];
      snprintf (origprocname, sizeof (origprocname), procpath, fd);
      char *origlink = (char *) alloca (PATH_MAX);
      ssize_t n = readlink (origprocname, origlink, PATH_MAX - 1);
      if (n == -1)
	goto no_debuginfo;
      origlink[n] = '\0';

      /* Try to find the actual file.  There are three places:
	 1. the same directory the DSO is in
	 2. in a subdir named .debug of the directory the DSO is in
	 3. in /usr/lib/debug/PATH-OF-DSO
      */
      char *realname = canonicalize_file_name (origlink);
      char *cp = NULL;
      if (realname == NULL || (cp = strrchr (realname, '/')) == NULL)
	error (EXIT_FAILURE, errno, _("cannot determine file name"));

      /* Leave the last slash in place.  */
      *++cp = '\0';

      /* First add the debuginfo file name only.  */
      static const char usrlibdebug[]= "/usr/lib/debug/";
      char *workbuf = (char *) alloca (sizeof (usrlibdebug)
				       + (cp - realname)
				       + strlen (debuginfo_fname));
      strcpy (stpcpy (workbuf, realname), debuginfo_fname);

      int fd2 = open (workbuf, O_RDONLY);
      if (fd2 == -1)
	{
	  strcpy (stpcpy (stpcpy (workbuf, realname), ".debug/"),
		  debuginfo_fname);
	  fd2 = open (workbuf, O_RDONLY);
	  if (fd2 == -1)
	    {
	      strcpy (stpcpy (stpcpy (workbuf, usrlibdebug), realname),
		      debuginfo_fname);
	      fd2 = open (workbuf, O_RDONLY);
	    }
	}

      if (fd2 != -1)
	{
	  ElfW(Ehdr) ehdr2;

	  /* Read the ELF header.  */
	  if (pread (fd2, &ehdr2, sizeof (ehdr2), 0) != sizeof (ehdr2))
	    error (EXIT_FAILURE, errno,
		   _("reading of ELF header failed"));

	  /* Map the section header.  */
	  size_t size = ehdr2.e_shnum * sizeof (ElfW(Shdr));
	  ElfW(Shdr) *shdr2 = (ElfW(Shdr) *) alloca (size);
	  if (pread (fd2, shdr2, size, ehdr2.e_shoff) != size)
	    error (EXIT_FAILURE, errno,
		   _("reading of section headers failed"));

	  /* Get the section header string table.  */
	  shstrtab = (char *) alloca (shdr2[ehdr2.e_shstrndx].sh_size);
	  if (pread (fd2, shstrtab, shdr2[ehdr2.e_shstrndx].sh_size,
		     shdr2[ehdr2.e_shstrndx].sh_offset)
	      != shdr2[ehdr2.e_shstrndx].sh_size)
	    error (EXIT_FAILURE, errno,
		   _("reading of section header string table failed"));

	  /* Search for the ".symtab" section.  */
	  for (int idx = 0; idx < ehdr2.e_shnum; ++idx)
	    if (shdr2[idx].sh_type == SHT_SYMTAB
		&& strcmp (shstrtab + shdr2[idx].sh_name, ".symtab") == 0)
	      {
		symtab_entry = &shdr2[idx];
		shdr = shdr2;
		symfd = fd2;
		break;
	      }

	  if  (fd2 != symfd)
	    close (fd2);
	}
    }

 no_debuginfo:
  if (symtab_entry == NULL)
    {
      fprintf (stderr, _("\
*** The file `%s' is stripped: no detailed analysis possible\n"),
	      name);
      result->symtab = NULL;
      result->strtab = NULL;
    }
  else
    {
      ElfW(Off) min_offset, max_offset;
      ElfW(Shdr) *strtab_entry;

      strtab_entry = &shdr[symtab_entry->sh_link];

      /* Find the minimum and maximum offsets that include both the symbol
	 table and the string table.  */
      if (symtab_entry->sh_offset < strtab_entry->sh_offset)
	{
	  min_offset = symtab_entry->sh_offset & ~(pagesize - 1);
	  max_offset = strtab_entry->sh_offset + strtab_entry->sh_size;
	}
      else
	{
	  min_offset = strtab_entry->sh_offset & ~(pagesize - 1);
	  max_offset = symtab_entry->sh_offset + symtab_entry->sh_size;
	}

      result->symbol_map = mmap (NULL, max_offset - min_offset,
				 PROT_READ, MAP_SHARED|MAP_FILE, symfd,
				 min_offset);
      if (result->symbol_map == MAP_FAILED)
	error (EXIT_FAILURE, errno, _("failed to load symbol data"));

      result->symtab
	= (const ElfW(Sym) *) ((const char *) result->symbol_map
			       + (symtab_entry->sh_offset - min_offset));
      result->symtab_size = symtab_entry->sh_size;
      result->strtab = ((const char *) result->symbol_map
			+ (strtab_entry->sh_offset - min_offset));
      result->symbol_mapsize = max_offset - min_offset;
    }

  /* Free the descriptor for the shared object.  */
  close (fd);
  if (symfd != fd)
    close (symfd);

  return result;
}


static void
unload_shobj (struct shobj *shobj)
{
  munmap (shobj->symbol_map, shobj->symbol_mapsize);
  dlclose (shobj->map);
}


static struct profdata *
load_profdata (const char *name, struct shobj *shobj)
{
  struct profdata *result;
  int fd;
  struct stat64 st;
  void *addr;
  uint32_t *narcsp;
  size_t fromlimit;
  struct here_cg_arc_record *data;
  struct here_fromstruct *froms;
  uint16_t *tos;
  size_t fromidx;
  size_t idx;

  fd = open (name, O_RDONLY);
  if (fd == -1)
    {
      char *ext_name;

      if (errno != ENOENT || strchr (name, '/') != NULL)
	/* The file exists but we are not allowed to read it or the
	   file does not exist and the name includes a path
	   specification..  */
	return NULL;

      /* A file with the given name does not exist in the current
	 directory, try it in the default location where the profiling
	 files are created.  */
      ext_name = (char *) alloca (strlen (name) + sizeof "/var/tmp/");
      stpcpy (stpcpy (ext_name, "/var/tmp/"), name);
      name = ext_name;

      fd = open (ext_name, O_RDONLY);
      if (fd == -1)
	{
	  /* Even this file does not exist.  */
	  error (0, errno, _("cannot load profiling data"));
	  return NULL;
	}
    }

  /* We have found the file, now make sure it is the right one for the
     data file.  */
  if (fstat64 (fd, &st) < 0)
    {
      error (0, errno, _("while stat'ing profiling data file"));
      close (fd);
      return NULL;
    }

  if ((size_t) st.st_size != shobj->expected_size)
    {
      error (0, 0,
	     _("profiling data file `%s' does not match shared object `%s'"),
	     name, shobj->name);
      close (fd);
      return NULL;
    }

  /* The data file is most probably the right one for our shared
     object.  Map it now.  */
  addr = mmap (NULL, st.st_size, PROT_READ, MAP_SHARED|MAP_FILE, fd, 0);
  if (addr == MAP_FAILED)
    {
      error (0, errno, _("failed to mmap the profiling data file"));
      close (fd);
      return NULL;
    }

  /* We don't need the file desriptor anymore.  */
  if (close (fd) < 0)
    {
      error (0, errno, _("error while closing the profiling data file"));
      munmap (addr, st.st_size);
      return NULL;
    }

  /* Prepare the result.  */
  result = (struct profdata *) calloc (1, sizeof (struct profdata));
  if (result == NULL)
    {
      error (0, errno, _("cannot create internal descriptor"));
      munmap (addr, st.st_size);
      return NULL;
    }

  /* Store the address and size so that we can later free the resources.  */
  result->addr = addr;
  result->size = st.st_size;

  /* Pointer to data after the header.  */
  result->hist = (char *) ((struct gmon_hdr *) addr + 1);
  result->hist_hdr = (struct real_gmon_hist_hdr *) ((char *) result->hist
						    + sizeof (uint32_t));
  result->kcount = (uint16_t *) ((char *) result->hist + sizeof (uint32_t)
				 + sizeof (struct real_gmon_hist_hdr));

  /* Compute pointer to array of the arc information.  */
  narcsp = (uint32_t *) ((char *) result->kcount + shobj->kcountsize
			 + sizeof (uint32_t));
  result->narcs = *narcsp;
  result->data = (struct here_cg_arc_record *) ((char *) narcsp
						+ sizeof (uint32_t));

  /* Create the gmon_hdr we expect or write.  */
  struct real_gmon_hdr
  {
    char cookie[4];
    int32_t version;
    char spare[3 * 4];
  } gmon_hdr;
  if (sizeof (gmon_hdr) != sizeof (struct gmon_hdr)
      || (offsetof (struct real_gmon_hdr, cookie)
	  != offsetof (struct gmon_hdr, cookie))
      || (offsetof (struct real_gmon_hdr, version)
	  != offsetof (struct gmon_hdr, version)))
    abort ();

  memcpy (&gmon_hdr.cookie[0], GMON_MAGIC, sizeof (gmon_hdr.cookie));
  gmon_hdr.version = GMON_SHOBJ_VERSION;
  memset (gmon_hdr.spare, '\0', sizeof (gmon_hdr.spare));

  /* Create the hist_hdr we expect or write.  */
  struct real_gmon_hist_hdr hist_hdr;
  if (sizeof (hist_hdr) != sizeof (struct gmon_hist_hdr)
      || (offsetof (struct real_gmon_hist_hdr, low_pc)
	  != offsetof (struct gmon_hist_hdr, low_pc))
      || (offsetof (struct real_gmon_hist_hdr, high_pc)
	  != offsetof (struct gmon_hist_hdr, high_pc))
      || (offsetof (struct real_gmon_hist_hdr, hist_size)
	  != offsetof (struct gmon_hist_hdr, hist_size))
      || (offsetof (struct real_gmon_hist_hdr, prof_rate)
	  != offsetof (struct gmon_hist_hdr, prof_rate))
      || (offsetof (struct real_gmon_hist_hdr, dimen)
	  != offsetof (struct gmon_hist_hdr, dimen))
      || (offsetof (struct real_gmon_hist_hdr, dimen_abbrev)
	  != offsetof (struct gmon_hist_hdr, dimen_abbrev)))
    abort ();

  hist_hdr.low_pc = (char *) shobj->lowpc - shobj->map->l_addr;
  hist_hdr.high_pc = (char *) shobj->highpc - shobj->map->l_addr;
  if (do_test)
    printf ("low_pc = %p\nhigh_pc = %p\n", hist_hdr.low_pc, hist_hdr.high_pc);
  hist_hdr.hist_size = shobj->kcountsize / sizeof (HISTCOUNTER);
  hist_hdr.prof_rate = __profile_frequency ();
  strncpy (hist_hdr.dimen, "seconds", sizeof (hist_hdr.dimen));
  hist_hdr.dimen_abbrev = 's';

  /* Test whether the header of the profiling data is ok.  */
  if (memcmp (addr, &gmon_hdr, sizeof (struct gmon_hdr)) != 0
      || *(uint32_t *) result->hist != GMON_TAG_TIME_HIST
      || memcmp (result->hist_hdr, &hist_hdr,
		 sizeof (struct gmon_hist_hdr)) != 0
      || narcsp[-1] != GMON_TAG_CG_ARC)
    {
      error (0, 0, _("`%s' is no correct profile data file for `%s'"),
	     name, shobj->name);
      if (do_test)
	{
	  if (memcmp (addr, &gmon_hdr, sizeof (struct gmon_hdr)) != 0)
	    puts ("gmon_hdr differs");
	  if (*(uint32_t *) result->hist != GMON_TAG_TIME_HIST)
	    puts ("result->hist differs");
	  if (memcmp (result->hist_hdr, &hist_hdr,
		      sizeof (struct gmon_hist_hdr)) != 0)
	    puts ("hist_hdr differs");
	  if (narcsp[-1] != GMON_TAG_CG_ARC)
	    puts ("narcsp[-1] differs");
	}
      free (result);
      munmap (addr, st.st_size);
      return NULL;
    }

  /* We are pretty sure now that this is a correct input file.  Set up
     the remaining information in the result structure and return.  */
  result->tos = (uint16_t *) calloc (shobj->tossize + shobj->fromssize, 1);
  if (result->tos == NULL)
    {
      error (0, errno, _("cannot create internal descriptor"));
      munmap (addr, st.st_size);
      free (result);
      return NULL;
    }

  result->froms = (struct here_fromstruct *) ((char *) result->tos
					      + shobj->tossize);
  fromidx = 0;

  /* Now we have to process all the arc count entries.  */
  fromlimit = shobj->fromlimit;
  data = result->data;
  froms = result->froms;
  tos = result->tos;
  for (idx = 0; idx < MIN (*narcsp, fromlimit); ++idx)
    {
      size_t to_index;
      size_t newfromidx;
      to_index = (data[idx].self_pc / (shobj->hashfraction * sizeof (*tos)));
      newfromidx = fromidx++;
      froms[newfromidx].here = &data[idx];
      froms[newfromidx].link = tos[to_index];
      tos[to_index] = newfromidx;
    }

  return result;
}


static void
unload_profdata (struct profdata *profdata)
{
  free (profdata->tos);
  munmap (profdata->addr, profdata->size);
  free (profdata);
}


static void
count_total_ticks (struct shobj *shobj, struct profdata *profdata)
{
  volatile uint16_t *kcount = profdata->kcount;
  size_t maxkidx = shobj->kcountsize;
  size_t factor = 2 * (65536 / shobj->s_scale);
  size_t kidx = 0;
  size_t sidx = 0;

  while (sidx < symidx)
    {
      uintptr_t start = sortsym[sidx]->addr;
      uintptr_t end = start + sortsym[sidx]->size;

      while (kidx < maxkidx && factor * kidx < start)
	++kidx;
      if (kidx == maxkidx)
	break;

      while (kidx < maxkidx && factor * kidx < end)
	sortsym[sidx]->ticks += kcount[kidx++];
      if (kidx == maxkidx)
	break;

      total_ticks += sortsym[sidx++]->ticks;
    }
}


static size_t
find_symbol (uintptr_t addr)
{
  size_t sidx = 0;

  while (sidx < symidx)
    {
      uintptr_t start = sortsym[sidx]->addr;
      uintptr_t end = start + sortsym[sidx]->size;

      if (addr >= start && addr < end)
	return sidx;

      if (addr < start)
	break;

      ++sidx;
    }

  return (size_t) -1l;
}


static void
count_calls (struct shobj *shobj, struct profdata *profdata)
{
  struct here_cg_arc_record *data = profdata->data;
  uint32_t narcs = profdata->narcs;
  uint32_t cnt;

  for (cnt = 0; cnt < narcs; ++cnt)
    {
      uintptr_t here = data[cnt].self_pc;
      size_t symbol_idx;

      /* Find the symbol for this address.  */
      symbol_idx = find_symbol (here);
      if (symbol_idx != (size_t) -1l)
	sortsym[symbol_idx]->calls += data[cnt].count;
    }
}


static int
symorder (const void *o1, const void *o2)
{
  const struct known_symbol *p1 = (const struct known_symbol *) o1;
  const struct known_symbol *p2 = (const struct known_symbol *) o2;

  return p1->addr - p2->addr;
}


static void
printsym (const void *node, VISIT value, int level)
{
  if (value == leaf || value == postorder)
    sortsym[symidx++] = *(struct known_symbol **) node;
}


static void
read_symbols (struct shobj *shobj)
{
  int n = 0;

  /* Initialize the obstacks.  */
#define obstack_chunk_alloc malloc
#define obstack_chunk_free free
  obstack_init (&shobj->ob_str);
  obstack_init (&shobj->ob_sym);
  obstack_init (&ob_list);

  /* Process the symbols.  */
  if (shobj->symtab != NULL)
    {
      const ElfW(Sym) *sym = shobj->symtab;
      const ElfW(Sym) *sym_end
	= (const ElfW(Sym) *) ((const char *) sym + shobj->symtab_size);
      for (; sym < sym_end; sym++)
	if ((ELFW(ST_TYPE) (sym->st_info) == STT_FUNC
	     || ELFW(ST_TYPE) (sym->st_info) == STT_NOTYPE)
	    && sym->st_size != 0)
	  {
	    struct known_symbol **existp;
	    struct known_symbol *newsym
	      = (struct known_symbol *) obstack_alloc (&shobj->ob_sym,
						       sizeof (*newsym));
	    if (newsym == NULL)
	      error (EXIT_FAILURE, errno, _("cannot allocate symbol data"));

	    newsym->name = &shobj->strtab[sym->st_name];
	    newsym->addr = sym->st_value;
	    newsym->size = sym->st_size;
	    newsym->weak = ELFW(ST_BIND) (sym->st_info) == STB_WEAK;
	    newsym->hidden = (ELFW(ST_VISIBILITY) (sym->st_other)
			      != STV_DEFAULT);
	    newsym->ticks = 0;
	    newsym->calls = 0;

	    existp = tfind (newsym, &symroot, symorder);
	    if (existp == NULL)
	      {
		/* New function.  */
		tsearch (newsym, &symroot, symorder);
		++n;
	      }
	    else
	      {
		/* The function is already defined.  See whether we have
		   a better name here.  */
		if (((*existp)->hidden && !newsym->hidden)
		    || ((*existp)->name[0] == '_' && newsym->name[0] != '_')
		    || ((*existp)->name[0] != '_' && newsym->name[0] != '_'
			&& ((*existp)->weak && !newsym->weak)))
		  *existp = newsym;
		else
		  /* We don't need the allocated memory.  */
		  obstack_free (&shobj->ob_sym, newsym);
	      }
	  }
    }
  else
    {
      /* Blarg, the binary is stripped.  We have to rely on the
	 information contained in the dynamic section of the object.  */
      const ElfW(Sym) *symtab = (ElfW(Sym) *) D_PTR (shobj->map,
						     l_info[DT_SYMTAB]);
      const char *strtab = (const char *) D_PTR (shobj->map,
						 l_info[DT_STRTAB]);

      /* We assume that the string table follows the symbol table,
	 because there is no way in ELF to know the size of the
	 dynamic symbol table without looking at the section headers.  */
      while ((void *) symtab < (void *) strtab)
	{
	  if ((ELFW(ST_TYPE)(symtab->st_info) == STT_FUNC
	       || ELFW(ST_TYPE)(symtab->st_info) == STT_NOTYPE)
	      && symtab->st_size != 0)
	    {
	      struct known_symbol *newsym;
	      struct known_symbol **existp;

	      newsym =
		(struct known_symbol *) obstack_alloc (&shobj->ob_sym,
						       sizeof (*newsym));
	      if (newsym == NULL)
		error (EXIT_FAILURE, errno, _("cannot allocate symbol data"));

	      newsym->name = &strtab[symtab->st_name];
	      newsym->addr = symtab->st_value;
	      newsym->size = symtab->st_size;
	      newsym->weak = ELFW(ST_BIND) (symtab->st_info) == STB_WEAK;
	      newsym->hidden = (ELFW(ST_VISIBILITY) (symtab->st_other)
				!= STV_DEFAULT);
	      newsym->ticks = 0;
	      newsym->froms = NULL;
	      newsym->tos = NULL;

	      existp = tfind (newsym, &symroot, symorder);
	      if (existp == NULL)
		{
		  /* New function.  */
		  tsearch (newsym, &symroot, symorder);
		  ++n;
		}
	      else
		{
		  /* The function is already defined.  See whether we have
		     a better name here.  */
		  if (((*existp)->hidden && !newsym->hidden)
		      || ((*existp)->name[0] == '_' && newsym->name[0] != '_')
		      || ((*existp)->name[0] != '_' && newsym->name[0] != '_'
			  && ((*existp)->weak && !newsym->weak)))
		    *existp = newsym;
		  else
		    /* We don't need the allocated memory.  */
		    obstack_free (&shobj->ob_sym, newsym);
		}
	    }

	  ++symtab;
	}
    }

  sortsym = malloc (n * sizeof (struct known_symbol *));
  if (sortsym == NULL)
    abort ();

  twalk (symroot, printsym);
}


static void
add_arcs (struct profdata *profdata)
{
  uint32_t narcs = profdata->narcs;
  struct here_cg_arc_record *data = profdata->data;
  uint32_t cnt;

  for (cnt = 0; cnt < narcs; ++cnt)
    {
      /* First add the incoming arc.  */
      size_t sym_idx = find_symbol (data[cnt].self_pc);

      if (sym_idx != (size_t) -1l)
	{
	  struct known_symbol *sym = sortsym[sym_idx];
	  struct arc_list *runp = sym->froms;

	  while (runp != NULL
		 && ((data[cnt].from_pc == 0 && runp->idx != (size_t) -1l)
		     || (data[cnt].from_pc != 0
			 && (runp->idx == (size_t) -1l
			     || data[cnt].from_pc < sortsym[runp->idx]->addr
			     || (data[cnt].from_pc
				 >= (sortsym[runp->idx]->addr
				     + sortsym[runp->idx]->size))))))
	    runp = runp->next;

	  if (runp == NULL)
	    {
	      /* We need a new entry.  */
	      struct arc_list *newp = (struct arc_list *)
		obstack_alloc (&ob_list, sizeof (struct arc_list));

	      if (data[cnt].from_pc == 0)
		newp->idx = (size_t) -1l;
	      else
		newp->idx = find_symbol (data[cnt].from_pc);
	      newp->count = data[cnt].count;
	      newp->next = sym->froms;
	      sym->froms = newp;
	    }
	  else
	    /* Increment the counter for the found entry.  */
	    runp->count += data[cnt].count;
	}

      /* Now add it to the appropriate outgoing list.  */
      sym_idx = find_symbol (data[cnt].from_pc);
      if (sym_idx != (size_t) -1l)
	{
	  struct known_symbol *sym = sortsym[sym_idx];
	  struct arc_list *runp = sym->tos;

	  while (runp != NULL
		 && (runp->idx == (size_t) -1l
		     || data[cnt].self_pc < sortsym[runp->idx]->addr
		     || data[cnt].self_pc >= (sortsym[runp->idx]->addr
					      + sortsym[runp->idx]->size)))
	    runp = runp->next;

	  if (runp == NULL)
	    {
	      /* We need a new entry.  */
	      struct arc_list *newp = (struct arc_list *)
		obstack_alloc (&ob_list, sizeof (struct arc_list));

	      newp->idx = find_symbol (data[cnt].self_pc);
	      newp->count = data[cnt].count;
	      newp->next = sym->tos;
	      sym->tos = newp;
	    }
	  else
	    /* Increment the counter for the found entry.  */
	    runp->count += data[cnt].count;
	}
    }
}


static int
countorder (const void *p1, const void *p2)
{
  struct known_symbol *s1 = (struct known_symbol *) p1;
  struct known_symbol *s2 = (struct known_symbol *) p2;

  if (s1->ticks != s2->ticks)
    return (int) (s2->ticks - s1->ticks);

  if (s1->calls != s2->calls)
    return (int) (s2->calls - s1->calls);

  return strcmp (s1->name, s2->name);
}


static double tick_unit;
static uintmax_t cumu_ticks;

static void
printflat (const void *node, VISIT value, int level)
{
  if (value == leaf || value == postorder)
    {
      struct known_symbol *s = *(struct known_symbol **) node;

      cumu_ticks += s->ticks;

      printf ("%6.2f%10.2f%9.2f%9" PRIdMAX "%9.2f           %s\n",
	      total_ticks ? (100.0 * s->ticks) / total_ticks : 0.0,
	      tick_unit * cumu_ticks,
	      tick_unit * s->ticks,
	      s->calls,
	      s->calls ? (s->ticks * 1000000) * tick_unit / s->calls : 0,
	      /* FIXME: don't know about called functions.  */
	      s->name);
    }
}


/* ARGUSED */
static void
freenoop (void *p)
{
}


static void
generate_flat_profile (struct profdata *profdata)
{
  size_t n;
  void *data = NULL;

  tick_unit = 1.0 / profdata->hist_hdr->prof_rate;

  printf ("Flat profile:\n\n"
	  "Each sample counts as %g %s.\n",
	  tick_unit, profdata->hist_hdr->dimen);
  fputs ("  %   cumulative   self              self     total\n"
	 " time   seconds   seconds    calls  us/call  us/call  name\n",
	 stdout);

  for (n = 0; n < symidx; ++n)
    if (sortsym[n]->calls != 0 || sortsym[n]->ticks != 0)
      tsearch (sortsym[n], &data, countorder);

  twalk (data, printflat);

  tdestroy (data, freenoop);
}


static void
generate_call_graph (struct profdata *profdata)
{
  size_t cnt;

  puts ("\nindex % time    self  children    called     name\n");

  for (cnt = 0; cnt < symidx; ++cnt)
    if (sortsym[cnt]->froms != NULL || sortsym[cnt]->tos != NULL)
      {
	struct arc_list *runp;
	size_t n;

	/* First print the from-information.  */
	runp = sortsym[cnt]->froms;
	while (runp != NULL)
	  {
	    printf ("            %8.2f%8.2f%9" PRIdMAX "/%-9" PRIdMAX "   %s",
		    (runp->idx != (size_t) -1l
		     ? sortsym[runp->idx]->ticks * tick_unit : 0.0),
		    0.0, /* FIXME: what's time for the children, recursive */
		    runp->count, sortsym[cnt]->calls,
		    (runp->idx != (size_t) -1l
		     ? sortsym[runp->idx]->name : "<UNKNOWN>"));

	    if (runp->idx != (size_t) -1l)
	      printf (" [%zd]", runp->idx);
	    putchar_unlocked ('\n');

	    runp = runp->next;
	  }

	/* Info about the function itself.  */
	n = printf ("[%zu]", cnt);
	printf ("%*s%5.1f%8.2f%8.2f%9" PRIdMAX "         %s [%zd]\n",
		(int) (7 - n), " ",
		total_ticks ? (100.0 * sortsym[cnt]->ticks) / total_ticks : 0,
		sortsym[cnt]->ticks * tick_unit,
		0.0, /* FIXME: what's time for the children, recursive */
		sortsym[cnt]->calls,
		sortsym[cnt]->name, cnt);

	/* Info about the functions this function calls.  */
	runp = sortsym[cnt]->tos;
	while (runp != NULL)
	  {
	    printf ("            %8.2f%8.2f%9" PRIdMAX "/",
		    (runp->idx != (size_t) -1l
		     ? sortsym[runp->idx]->ticks * tick_unit : 0.0),
		    0.0, /* FIXME: what's time for the children, recursive */
		    runp->count);

	    if (runp->idx != (size_t) -1l)
	      printf ("%-9" PRIdMAX "   %s [%zd]\n",
		      sortsym[runp->idx]->calls,
		      sortsym[runp->idx]->name,
		      runp->idx);
	    else
	      fputs ("???         <UNKNOWN>\n\n", stdout);

	    runp = runp->next;
	  }

	fputs ("-----------------------------------------------\n", stdout);
      }
}


static void
generate_call_pair_list (struct profdata *profdata)
{
  size_t cnt;

  for (cnt = 0; cnt < symidx; ++cnt)
    if (sortsym[cnt]->froms != NULL || sortsym[cnt]->tos != NULL)
      {
	struct arc_list *runp;

	/* First print the incoming arcs.  */
	runp = sortsym[cnt]->froms;
	while (runp != NULL)
	  {
	    if (runp->idx == (size_t) -1l)
	      printf ("\
<UNKNOWN>                          %-34s %9" PRIdMAX "\n",
		      sortsym[cnt]->name, runp->count);
	    runp = runp->next;
	  }

	/* Next the outgoing arcs.  */
	runp = sortsym[cnt]->tos;
	while (runp != NULL)
	  {
	    printf ("%-34s %-34s %9" PRIdMAX "\n",
		    sortsym[cnt]->name,
		    (runp->idx != (size_t) -1l
		     ? sortsym[runp->idx]->name : "<UNKNOWN>"),
		    runp->count);
	    runp = runp->next;
	  }
      }
}
