/* Smoke testing GDB process attach with thread-local variable access.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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

/* This test runs GDB against a forked copy of itself, to check
   whether libthread_db can be loaded, and that access to thread-local
   variables works.  */

#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <support/capture_subprocess.h>
#include <support/check.h>
#include <support/xptrace.h>
#include <support/subprocess.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/test-driver.h>
#include <support/xstdio.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <unistd.h>

/* Starts out as zero, changed to 1 or 2 by the debugger, depending on
   the thread.  */
__thread volatile int altered_by_debugger;

/* Common prefix between 32-bit and 64-bit ELF.  */
struct elf_prefix
{
  unsigned char e_ident[EI_NIDENT];
  uint16_t e_type;
  uint16_t e_machine;
  uint32_t e_version;
};
_Static_assert (sizeof (struct elf_prefix) == EI_NIDENT + 8,
                "padding in struct elf_prefix");

/* Reads the ELF header from PATH.  Returns true if the header can be
   read, false if the file is too short.  */
static bool
read_elf_header (const char *path, struct elf_prefix *elf)
{
  int fd = xopen (path, O_RDONLY, 0);
  bool result = read (fd, elf, sizeof (*elf)) == sizeof (*elf);
  xclose (fd);
  return result;
}

/* Searches for "gdb" alongside the path variable.  See execvpe.  */
static char *
find_gdb (void)
{
  const char *path = getenv ("PATH");
  if (path == NULL)
    return NULL;
  while (true)
    {
      const char *colon = strchrnul (path, ':');
      char *candidate = xasprintf ("%.*s/gdb", (int) (colon - path), path);
      if (access (candidate, X_OK) == 0)
        return candidate;
      free (candidate);
      if (*colon == '\0')
        break;
      path = colon + 1;
    }
  return NULL;
}

/* Writes the GDB script to run the test to PATH.  */
static void
write_gdbscript (const char *path, int tested_pid)
{
  FILE *fp = xfopen (path, "w");
  fprintf (fp,
           "set trace-commands on\n"
           "set debug libthread-db 1\n"
#if DO_ADD_SYMBOL_FILE
           /* Do not do this unconditionally to work around a GDB
              assertion failure: ../../gdb/symtab.c:6404:
              internal-error: CORE_ADDR get_msymbol_address(objfile*,
              const minimal_symbol*): Assertion `(objf->flags &
              OBJF_MAINLINE) == 0' failed.  */
           "add-symbol-file %1$s/nptl/tst-pthread-gdb-attach\n"
#endif
           "set auto-load safe-path %1$s/nptl_db\n"
           "set libthread-db-search-path %1$s/nptl_db\n"
           "attach %2$d\n",
           support_objdir_root, tested_pid);
  fputs ("break debugger_inspection_point\n"
         "continue\n"
         "thread 1\n"
         "print altered_by_debugger\n"
         "print altered_by_debugger = 1\n"
         "thread 2\n"
         "print altered_by_debugger\n"
         "print altered_by_debugger = 2\n"
         "continue\n",
         fp);
  xfclose (fp);
}

/* The test sets a breakpoint on this function and alters the
   altered_by_debugger thread-local variable.  */
void __attribute__ ((weak))
debugger_inspection_point (void)
{
}

/* Thread function for the test thread in the subprocess.  */
static void *
subprocess_thread (void *closure)
{
  /* Wait until altered_by_debugger changes the value away from 0.  */
  while (altered_by_debugger == 0)
    {
      usleep (100 * 1000);
      debugger_inspection_point ();
    }

  TEST_COMPARE (altered_by_debugger, 2);
  return NULL;
}

/* This function implements the subprocess under test.  It creates a
   second thread, waiting for its value to change to 2, and checks
   that the main thread also changed its value to 1.  */
static void
in_subprocess (void *arg)
{
  pthread_t thr = xpthread_create (NULL, subprocess_thread, NULL);
  TEST_VERIFY (xpthread_join (thr) == NULL);
  TEST_COMPARE (altered_by_debugger, 1);
  _exit (0);
}

static void
gdb_process (const char *gdb_path, const char *gdbscript, pid_t *tested_pid)
{
  /* Create a copy of current test to check with gdb.  As the
     target_process is a child of this gdb_process, gdb is also able
     to attach to target_process if YAMA is configured to 1 =
     "restricted ptrace".  */
  struct support_subprocess target = support_subprocess (in_subprocess, NULL);

  write_gdbscript (gdbscript, target.pid);
  *tested_pid = target.pid;

  xdup2 (STDOUT_FILENO, STDERR_FILENO);
  execl (gdb_path, "gdb", "-nx", "-batch", "-x", gdbscript, NULL);
  if (errno == ENOENT)
    _exit (EXIT_UNSUPPORTED);
  else
    _exit (1);
}

static int
do_test (void)
{
  char *gdb_path = find_gdb ();
  if (gdb_path == NULL)
    FAIL_UNSUPPORTED ("gdb command not found in PATH: %s", getenv ("PATH"));

  /* Check that libthread_db is compatible with the gdb architecture
     because gdb loads it via dlopen.  */
  {
    char *threaddb_path = xasprintf ("%s/nptl_db/libthread_db.so",
                                     support_objdir_root);
    struct elf_prefix elf_threaddb;
    TEST_VERIFY_EXIT (read_elf_header (threaddb_path, &elf_threaddb));
    struct elf_prefix elf_gdb;
    /* If the ELF header cannot be read or "gdb" is not an ELF file,
       assume this is a wrapper script that can run.  */
    if (read_elf_header (gdb_path, &elf_gdb)
        && memcmp (&elf_gdb, ELFMAG, SELFMAG) == 0)
      {
        if (elf_gdb.e_ident[EI_CLASS] != elf_threaddb.e_ident[EI_CLASS])
          FAIL_UNSUPPORTED ("GDB at %s has wrong class", gdb_path);
        if (elf_gdb.e_ident[EI_DATA] != elf_threaddb.e_ident[EI_DATA])
          FAIL_UNSUPPORTED ("GDB at %s has wrong data", gdb_path);
        if (elf_gdb.e_machine != elf_threaddb.e_machine)
          FAIL_UNSUPPORTED ("GDB at %s has wrong machine", gdb_path);
      }
    free (threaddb_path);
  }

  /* Check if our subprocess can be debugged with ptrace.  */
  {
    int ptrace_scope = support_ptrace_scope ();
    if (ptrace_scope >= 2)
      FAIL_UNSUPPORTED ("/proc/sys/kernel/yama/ptrace_scope >= 2");
  }

  char *gdbscript;
  xclose (create_temp_file ("tst-pthread-gdb-attach-", &gdbscript));

  /* Run 'gdb' on test subprocess which will be created in gdb_process.
     The pid of the subprocess will be written to 'tested_pid'.  */
  pid_t *tested_pid = support_shared_allocate (sizeof (pid_t));

  pid_t gdb_pid = xfork ();
  if (gdb_pid == 0)
    gdb_process (gdb_path, gdbscript, tested_pid);

  int status;
  TEST_COMPARE (xwaitpid (gdb_pid, &status, 0), gdb_pid);
  if (WIFEXITED (status) && WEXITSTATUS (status) == EXIT_UNSUPPORTED)
    /* gdb is not installed.  */
    return EXIT_UNSUPPORTED;
  TEST_COMPARE (status, 0);

  kill (*tested_pid, SIGKILL);

  support_shared_free (tested_pid);
  free (gdbscript);
  free (gdb_path);
  return 0;
}

#include <support/test-driver.c>
