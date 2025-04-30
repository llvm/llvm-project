/* List dynamic shared objects linked into given process.
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

#define _FILE_OFFSET_BITS 64

#include <argp.h>
#include <dirent.h>
#include <error.h>
#include <fcntl.h>
#include <libintl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/ptrace.h>
#include <sys/wait.h>
#include <scratch_buffer.h>

#include <ldsodefs.h>
#include <version.h>

/* Global variables.  */
extern char *program_invocation_short_name;
#define PACKAGE _libc_intl_domainname

/* External functions.  */
#include <programs/xmalloc.h>

/* Name and version of program.  */
static void print_version (FILE *stream, struct argp_state *state);
void (*argp_program_version_hook) (FILE *, struct argp_state *) = print_version;

/* Function to print some extra text in the help message.  */
static char *more_help (int key, const char *text, void *input);

/* Definitions of arguments for argp functions.  */
static const struct argp_option options[] =
{
  { NULL, 0, NULL, 0, NULL }
};

/* Short description of program.  */
static const char doc[] = N_("\
List dynamic shared objects loaded into process.");

/* Strings for arguments in help texts.  */
static const char args_doc[] = N_("PID");

/* Prototype for option handler.  */
static error_t parse_opt (int key, char *arg, struct argp_state *state);

/* Data structure to communicate with argp functions.  */
static struct argp argp =
{
  options, parse_opt, args_doc, doc, NULL, more_help, NULL
};


/* Local functions.  */
static int get_process_info (const char *exe, int dfd, long int pid);
static void wait_for_ptrace_stop (long int pid);


int
main (int argc, char *argv[])
{
  /* Parse and process arguments.  */
  int remaining;
  argp_parse (&argp, argc, argv, 0, &remaining, NULL);

  if (remaining != argc - 1)
    {
      fprintf (stderr,
	       gettext ("Exactly one parameter with process ID required.\n"));
      argp_help (&argp, stderr, ARGP_HELP_SEE, program_invocation_short_name);
      return 1;
    }

  _Static_assert (sizeof (pid_t) == sizeof (int)
		  || sizeof (pid_t) == sizeof (long int),
		  "sizeof (pid_t) != sizeof (int) or sizeof (long int)");

  char *endp;
  errno = 0;
  long int pid = strtol (argv[remaining], &endp, 10);
  if (pid < 0 || (pid == ULONG_MAX && errno == ERANGE) || *endp != '\0'
      || (sizeof (pid_t) < sizeof (pid) && pid > INT_MAX))
    error (EXIT_FAILURE, 0, gettext ("invalid process ID '%s'"),
	   argv[remaining]);

  /* Determine the program name.  */
  char buf[7 + 3 * sizeof (pid)];
  snprintf (buf, sizeof (buf), "/proc/%lu", pid);
  int dfd = open (buf, O_RDONLY | O_DIRECTORY);
  if (dfd == -1)
    error (EXIT_FAILURE, errno, gettext ("cannot open %s"), buf);

  /* Name of the executable  */
  struct scratch_buffer exe;
  scratch_buffer_init (&exe);
  ssize_t nexe;
  while ((nexe = readlinkat (dfd, "exe",
			     exe.data, exe.length)) == exe.length)
    {
      if (!scratch_buffer_grow (&exe))
	{
	  nexe = -1;
	  break;
	}
    }
  if (nexe == -1)
    /* Default stack allocation is at least 1024.  */
    snprintf (exe.data, exe.length, "<program name undetermined>");
  else
    ((char*)exe.data)[nexe] = '\0';

  /* Stop all threads since otherwise the list of loaded modules might
     change while we are reading it.  */
  struct thread_list
  {
    pid_t tid;
    struct thread_list *next;
  } *thread_list = NULL;

  int taskfd = openat (dfd, "task", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (taskfd == 1)
    error (EXIT_FAILURE, errno, gettext ("cannot open %s/task"), buf);
  DIR *dir = fdopendir (taskfd);
  if (dir == NULL)
    error (EXIT_FAILURE, errno, gettext ("cannot prepare reading %s/task"),
	   buf);

  struct dirent *d;
  while ((d = readdir (dir)) != NULL)
    {
      if (! isdigit (d->d_name[0]))
	continue;

      errno = 0;
      long int tid = strtol (d->d_name, &endp, 10);
      if (tid < 0 || (tid == ULONG_MAX && errno == ERANGE) || *endp != '\0'
	  || (sizeof (pid_t) < sizeof (pid) && tid > INT_MAX))
	error (EXIT_FAILURE, 0, gettext ("invalid thread ID '%s'"),
	       d->d_name);

      if (ptrace (PTRACE_ATTACH, tid, NULL, NULL) != 0)
	{
	  /* There might be a race between reading the directory and
	     threads terminating.  Ignore errors attaching to unknown
	     threads unless this is the main thread.  */
	  if (errno == ESRCH && tid != pid)
	    continue;

	  error (EXIT_FAILURE, errno, gettext ("cannot attach to process %lu"),
		 tid);
	}

      wait_for_ptrace_stop (tid);

      struct thread_list *newp = xmalloc (sizeof (*newp));
      newp->tid = tid;
      newp->next = thread_list;
      thread_list = newp;
    }

  closedir (dir);

  if (thread_list == NULL)
    error (EXIT_FAILURE, 0, gettext ("no valid %s/task entries"), buf);

  int status = get_process_info (exe.data, dfd, pid);

  do
    {
      ptrace (PTRACE_DETACH, thread_list->tid, NULL, NULL);
      struct thread_list *prev = thread_list;
      thread_list = thread_list->next;
      free (prev);
    }
  while (thread_list != NULL);

  close (dfd);
  scratch_buffer_free (&exe);

  return status;
}


/* Wait for PID to enter ptrace-stop state after being attached.  */
static void
wait_for_ptrace_stop (long int pid)
{
  int status;

  /* While waiting for SIGSTOP being delivered to the tracee we have to
     reinject any other pending signal.  Ignore all other errors.  */
  while (waitpid (pid, &status, __WALL) == pid && WIFSTOPPED (status))
    {
      /* The STOP signal should not be delivered to the tracee.  */
      if (WSTOPSIG (status) == SIGSTOP)
	return;
      if (ptrace (PTRACE_CONT, pid, NULL,
		  (void *) (uintptr_t) WSTOPSIG (status)))
	/* The only possible error is that the process died.  */
	return;
    }
}


/* Handle program arguments.  */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  switch (key)
    {
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}


/* Print bug-reporting information in the help message.  */
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
  fprintf (stream, "pldd %s%s\n", PKGVERSION, VERSION);
  fprintf (stream, gettext ("\
Copyright (C) %s Free Software Foundation, Inc.\n\
This is free software; see the source for copying conditions.  There is NO\n\
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n\
"), "2021");
  fprintf (stream, gettext ("Written by %s.\n"), "Ulrich Drepper");
}


#define CLASS 32
#include "pldd-xx.c"
#define CLASS 64
#include "pldd-xx.c"


static int
get_process_info (const char *exe, int dfd, long int pid)
{
  /* File descriptor of /proc/<pid>/mem file.  */
  int memfd = openat (dfd, "mem", O_RDONLY);
  if (memfd == -1)
    goto no_info;

  int fd = openat (dfd, "exe", O_RDONLY);
  if (fd == -1)
    {
    no_info:
      error (0, errno, gettext ("cannot get information about process %lu"),
	     pid);
      return EXIT_FAILURE;
    }

  char e_ident[EI_NIDENT];
  if (read (fd, e_ident, EI_NIDENT) != EI_NIDENT)
    goto no_info;

  close (fd);

  if (memcmp (e_ident, ELFMAG, SELFMAG) != 0)
    {
      error (0, 0, gettext ("process %lu is no ELF program"), pid);
      return EXIT_FAILURE;
    }

  fd = openat (dfd, "auxv", O_RDONLY);
  if (fd == -1)
    goto no_info;

  size_t auxv_size = 0;
  void *auxv = NULL;
  while (1)
    {
      auxv_size += 512;
      auxv = xrealloc (auxv, auxv_size);

      ssize_t n = pread (fd, auxv, auxv_size, 0);
      if (n < 0)
	goto no_info;
      if (n < auxv_size)
	{
	  auxv_size = n;
	  break;
	}
    }

  close (fd);

  int retval;
  if (e_ident[EI_CLASS] == ELFCLASS32)
    retval = find_maps32 (exe, memfd, pid, auxv, auxv_size);
  else
    retval = find_maps64 (exe, memfd, pid, auxv, auxv_size);

  free (auxv);
  close (memfd);

  return retval;
}
