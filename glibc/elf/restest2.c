#include <sys/types.h>
#include <dlfcn.h>
#include <error.h>
#include <mcheck.h>
#include <stdlib.h>
#include <unistd.h>

pid_t pid, pid2;

pid_t getpid(void)
{
  pid_t (*f)(void);
  f = (pid_t (*)(void)) dlsym (RTLD_NEXT, "getpid");
  if (f == NULL)
    error (EXIT_FAILURE, 0, "dlsym (RTLD_NEXT, \"getpid\"): %s", dlerror ());
  return (pid2 = f()) + 26;
}

int
main (void)
{
  pid_t (*f)(void);

  mtrace ();

  f = (pid_t (*)(void)) dlsym (RTLD_DEFAULT, "getpid");
  if (f == NULL)
    error (EXIT_FAILURE, 0, "dlsym (RTLD_DEFAULT, \"getpid\"): %s", dlerror ());
  pid = f();
  if (pid != pid2 + 26)
    error (EXIT_FAILURE, 0, "main getpid() not called");
  return 0;
}
