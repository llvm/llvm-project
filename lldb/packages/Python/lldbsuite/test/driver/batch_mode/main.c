#include <stdio.h>
#include <string.h>
#include <unistd.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

int 
main (int argc, char **argv)
{
#if defined(__linux__)
        // Immediately enable any ptracer so that we can allow the stub attach
        // operation to succeed.  Some Linux kernels are locked down so that
        // only an ancestor process can be a ptracer of a process.  This disables that
        // restriction.  Without it, attach-related stub tests will fail.
#if defined(PR_SET_PTRACER) && defined(PR_SET_PTRACER_ANY)
        // For now we execute on best effort basis.  If this fails for
        // some reason, so be it.
        const int prctl_result = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
        (void) prctl_result;
#endif
#endif

    int do_crash = 0;
    int do_wait = 0;

    int idx;
    for (idx = 1; idx < argc; idx++)
    {
        if (strcmp(argv[idx], "CRASH") == 0)
            do_crash = 1;
        if (strcmp(argv[idx], "WAIT") == 0)
            do_wait = 1;
    }
    printf("PID: %d END\n", getpid());

    if (do_wait)
    {
        int keep_waiting = 1;
        while (keep_waiting)
        {
            printf ("Waiting\n");
            sleep(1); // Stop here to unset keep_waiting
        }
    }

    if (do_crash)
    {
      char *touch_me_not = (char *) 0;
      printf ("About to crash.\n");
      touch_me_not[0] = 'a';
    }
    printf ("Got there on time and it did not crash.\n");
    return 0;
}
