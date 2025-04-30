/* Trace calls through PLTs and show caller, callee, and parameters.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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

#include <error.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/uio.h>

#include <ldsodefs.h>


extern const char *__progname;
extern const char *__progname_full;


/* List of objects to trace calls from.  */
static const char *fromlist;
/* List of objects to trace calls to.  */
static const char *tolist;

/* If non-zero, also trace returns of the calls.  */
static int do_exit;
/* If non-zero print PID for each line.  */
static int print_pid;

/* The output stream to use.  */
static FILE *out_file;


static int
match_pid (pid_t pid, const char *which)
{
  if (which == NULL || which[0] == '\0')
    {
      print_pid = 1;
      return 1;
    }

  char *endp;
  unsigned long n = strtoul (which, &endp, 0);
  return *endp == '\0' && n == pid;
}


static void
init (void)
{
  fromlist = getenv ("SOTRUSS_FROMLIST");
  if (fromlist != NULL && fromlist[0] == '\0')
    fromlist = NULL;
  tolist = getenv ("SOTRUSS_TOLIST");
  if (tolist != NULL && tolist[0] == '\0')
    tolist = NULL;
  do_exit = (getenv ("SOTRUSS_EXIT") ?: "")[0] != '\0';

  /* Determine whether this process is supposed to be traced and if
     yes, whether we should print into a file.  */
  const char *which_process = getenv ("SOTRUSS_WHICH");
  pid_t pid = getpid ();
  int out_fd = -1;
  if (match_pid (pid, which_process))
    {
      const char *out_filename = getenv ("SOTRUSS_OUTNAME");

      if (out_filename != NULL && out_filename[0] != 0)
	{
	  size_t out_filename_len = strlen (out_filename) + 13;
	  char fullname[out_filename_len];
	  char *endp = stpcpy (fullname, out_filename);
	  if (which_process == NULL || which_process[0] == '\0')
	    snprintf (endp, 13, ".%ld", (long int) pid);

	  out_fd = open (fullname, O_RDWR | O_CREAT | O_TRUNC, 0666);
	  if (out_fd != -1)
	    print_pid = 0;
	}
    }

  /* If we do not write into a file write to stderr.  Duplicate the
     descriptor so that we can keep printing in case the program
     closes stderr.  Try first to allocate a descriptor with a value
     usually not used as to minimize interference with the
     program.  */
  if (out_fd == -1)
    {
      out_fd = fcntl (STDERR_FILENO, F_DUPFD, 1000);
      if (out_fd == -1)
	out_fd = dup (STDERR_FILENO);
    }

  if (out_fd != -1)
    {
      /* Convert file descriptor into a stream.  */
      out_file = fdopen (out_fd, "w");
      if (out_file != NULL)
	setlinebuf (out_file);
    }
}


/* Audit interface verification.  We also initialize everything if
   everything checks out OK.  */
unsigned int
la_version (unsigned int v)
{
  if (v != LAV_CURRENT)
    error (1, 0, "cannot handle interface version %u", v);

  init ();

  return v;
}


/* Check whether a file name is on the colon-separated list of file
   names.  */
static unsigned int
match_file (const char *list, const char *name, size_t name_len,
	    unsigned int mask)
{
  if (list[0] == '\0')
    return 0;

  const char *cp = list;
  while (1)
    {
      if (strncmp (cp, name, name_len) == 0
	  && (cp[name_len] == ':' || cp[name_len] == '\0'))
	return mask;

      cp = strchr (cp, ':');
      if (cp == NULL)
	return 0;
      ++cp;
    }
}


unsigned int
la_objopen (struct link_map *map, Lmid_t lmid, uintptr_t *cookie)
{
  if (out_file == NULL)
    return 0;

  const char *full_name = map->l_name ?: "";
  if (full_name[0] == '\0')
    full_name = __progname_full;
  size_t full_name_len = strlen (full_name);
  const char *base_name = basename (full_name);
  if (base_name[0] == '\0')
    base_name = __progname;
  size_t base_name_len = strlen (base_name);

  int result = 0;
  const char *print_name = NULL;
  for (struct libname_list *l = map->l_libname; l != NULL; l = l->next)
    {
      if (print_name == NULL || (print_name[0] == '/' && l->name[0] != '/'))
	print_name = l->name;

      if (fromlist != NULL)
	result |= match_file (fromlist, l->name, strlen (l->name),
			      LA_FLG_BINDFROM);

      if (tolist != NULL)
	result |= match_file (tolist, l->name, strlen (l->name),LA_FLG_BINDTO);
    }

  if (print_name == NULL)
    print_name = base_name;
  if (print_name[0] == '\0')
    print_name = __progname;

  /* We cannot easily get to the object name in the PLT handling
     functions.  Use the cookie to get the string pointer passed back
     to us.  */
  *cookie = (uintptr_t) print_name;

  /* The object name has to be on the list of objects to trace calls
     from or that list must be empty.  In the latter case we trace
     only calls from the main binary.  */
  if (fromlist == NULL)
    result |= map->l_name[0] == '\0' ? LA_FLG_BINDFROM : 0;
  else
    result |= (match_file (fromlist, full_name, full_name_len,
			   LA_FLG_BINDFROM)
	       | match_file (fromlist, base_name, base_name_len,
			     LA_FLG_BINDFROM));

  /* The object name has to be on the list of objects to trace calls
     to or that list must be empty.  In the latter case we trace
     calls toall objects.  */
  if (tolist == NULL)
    result |= LA_FLG_BINDTO;
  else
    result |= (match_file (tolist, full_name, full_name_len, LA_FLG_BINDTO)
	       | match_file (tolist, base_name, base_name_len, LA_FLG_BINDTO));

  return result;
}


#if __ELF_NATIVE_CLASS == 32
# define la_symbind la_symbind32
typedef Elf32_Sym Elf_Sym;
#else
# define la_symbind la_symbind64
typedef Elf64_Sym Elf_Sym;
#endif

uintptr_t
la_symbind (Elf_Sym *sym, unsigned int ndx, uintptr_t *refcook,
	    uintptr_t *defcook, unsigned int *flags, const char *symname)
{
  if (!do_exit)
    *flags = LA_SYMB_NOPLTEXIT;

  return sym->st_value;
}


static void
print_enter (uintptr_t *refcook, uintptr_t *defcook, const char *symname,
	     unsigned long int reg1, unsigned long int reg2,
	     unsigned long int reg3, unsigned int flags)
{
  char buf[3 * sizeof (pid_t) + 3];
  buf[0] = '\0';
  if (print_pid)
    snprintf (buf, sizeof (buf), "%5ld: ", (long int) getpid ());

  fprintf (out_file, "%s%15s -> %-15s:%s%s(0x%lx, 0x%lx, 0x%lx)\n",
	   buf, (char *) *refcook, (char *) *defcook,
	   (flags & LA_SYMB_NOPLTEXIT) ? "*" : " ", symname, reg1, reg2, reg3);
}


#ifdef __i386__
Elf32_Addr
la_i86_gnu_pltenter (Elf32_Sym *sym __attribute__ ((unused)),
		     unsigned int ndx __attribute__ ((unused)),
		     uintptr_t *refcook, uintptr_t *defcook,
		     La_i86_regs *regs, unsigned int *flags,
		     const char *symname, long int *framesizep)
{
  unsigned long int *sp = (unsigned long int *) regs->lr_esp;

  print_enter (refcook, defcook, symname, sp[1], sp[2], sp[3], *flags);

  /* No need to copy anything, we will not need the parameters in any case.  */
  *framesizep = 0;

  return sym->st_value;
}
#elif defined __x86_64__
Elf64_Addr
la_x86_64_gnu_pltenter (Elf64_Sym *sym __attribute__ ((unused)),
			unsigned int ndx __attribute__ ((unused)),
			uintptr_t *refcook, uintptr_t *defcook,
			La_x86_64_regs *regs, unsigned int *flags,
			const char *symname, long int *framesizep)
{
  print_enter (refcook, defcook, symname,
	       regs->lr_rdi, regs->lr_rsi, regs->lr_rdx, *flags);

  /* No need to copy anything, we will not need the parameters in any case.  */
  *framesizep = 0;

  return sym->st_value;
}
#elif defined __sparc__ && !defined __arch64__
Elf32_Addr
la_sparc32_gnu_pltenter (Elf32_Sym *sym __attribute__ ((unused)),
			 unsigned int ndx __attribute__ ((unused)),
			 uintptr_t *refcook, uintptr_t *defcook,
			 La_sparc32_regs *regs, unsigned int *flags,
			 const char *symname, long int *framesizep)
{
  print_enter (refcook, defcook, symname,
	       regs->lr_reg[0], regs->lr_reg[1], regs->lr_reg[2],
	       *flags);

  /* No need to copy anything, we will not need the parameters in any case.  */
  *framesizep = 0;

  return sym->st_value;
}
#elif defined __sparc__ && defined __arch64__
Elf64_Addr
la_sparc64_gnu_pltenter (Elf64_Sym *sym __attribute__ ((unused)),
			 unsigned int ndx __attribute__ ((unused)),
			 uintptr_t *refcook, uintptr_t *defcook,
			 La_sparc64_regs *regs, unsigned int *flags,
			 const char *symname, long int *framesizep)
{
  print_enter (refcook, defcook, symname,
	       regs->lr_reg[0], regs->lr_reg[1], regs->lr_reg[2],
	       *flags);

  /* No need to copy anything, we will not need the parameters in any case.  */
  *framesizep = 0;

  return sym->st_value;
}
#elif !defined HAVE_ARCH_PLTENTER
# warning "pltenter for architecture not supported"
#endif


static void
print_exit (uintptr_t *refcook, uintptr_t *defcook, const char *symname,
	    unsigned long int reg)
{
  char buf[3 * sizeof (pid_t) + 3];
  buf[0] = '\0';
  if (print_pid)
    snprintf (buf, sizeof (buf), "%5ld: ", (long int) getpid ());

  fprintf (out_file, "%s%15s -> %-15s:%s%s - 0x%lx\n",
	   buf, (char *) *refcook, (char *) *defcook, " ", symname, reg);
}


#ifdef __i386__
unsigned int
la_i86_gnu_pltexit (Elf32_Sym *sym, unsigned int ndx, uintptr_t *refcook,
		    uintptr_t *defcook, const struct La_i86_regs *inregs,
		    struct La_i86_retval *outregs, const char *symname)
{
  print_exit (refcook, defcook, symname, outregs->lrv_eax);

  return 0;
}
#elif defined __x86_64__
unsigned int
la_x86_64_gnu_pltexit (Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
		       uintptr_t *defcook, const struct La_x86_64_regs *inregs,
		       struct La_x86_64_retval *outregs, const char *symname)
{
  print_exit (refcook, defcook, symname, outregs->lrv_rax);

  return 0;
}
#elif defined __sparc__ && !defined __arch64__
unsigned int
la_sparc32_gnu_pltexit (Elf32_Sym *sym, unsigned int ndx, uintptr_t *refcook,
			uintptr_t *defcook, const struct La_sparc32_regs *inregs,
			struct La_sparc32_retval *outregs, const char *symname)
{
  print_exit (refcook, defcook, symname, outregs->lrv_reg[0]);

  return 0;
}
#elif defined __sparc__ && defined __arch64__
unsigned int
la_sparc64_gnu_pltexit (Elf64_Sym *sym, unsigned int ndx, uintptr_t *refcook,
			uintptr_t *defcook, const struct La_sparc64_regs *inregs,
			struct La_sparc64_retval *outregs, const char *symname)
{
  print_exit (refcook, defcook, symname, outregs->lrv_reg[0]);

  return 0;
}
#elif !defined HAVE_ARCH_PLTEXIT
# warning "pltexit for architecture not supported"
#endif
