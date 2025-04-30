#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <error.h>
#include <errno.h>
#include <sys/wait.h>

void __attribute_noinline__ noop (void);

#define NR	2	/* Exit code of the child.  */

int
main (void)
{
  pid_t pid;
  int status;

  printf ("Before vfork\n");
  fflush (stdout);
  pid = vfork ();
  if (pid == 0)
    {
      /* This will clobber the return pc from vfork in the parent on
	 machines where it is stored on the stack, if vfork wasn't
	 implemented correctly, */
      noop ();
      _exit (NR);
    }
  else if (pid < 0)
    error (1, errno, "vfork");
  printf ("After vfork (parent)\n");
  if (waitpid (0, &status, 0) != pid
      || !WIFEXITED (status) || WEXITSTATUS (status) != NR)
    exit (1);

  return 0;
}

void
noop (void)
{
}
