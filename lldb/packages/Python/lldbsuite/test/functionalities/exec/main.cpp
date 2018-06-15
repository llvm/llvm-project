#include <errno.h>
#include <mach/mach.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <spawn.h>
#include <unistd.h>
#include <libgen.h>
#include <string>

static void
exit_with_errno (int err, const char *prefix)
{
    if (err)
    {
        fprintf (stderr,
                 "%s%s",
                 prefix ? prefix : "",
                 strerror(err));
        exit (err);
    }
}

static pid_t
spawn_process (const char *progname,
               const char **argv,
               const char **envp,
               int &err)
{
    pid_t pid = 0;

    const posix_spawn_file_actions_t *file_actions = NULL;
    posix_spawnattr_t attr;
    err = posix_spawnattr_init (&attr);
    if (err)
        return pid;

    short flags = POSIX_SPAWN_SETEXEC | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK;
    err = posix_spawnattr_setflags (&attr, flags);
    if (err == 0)
    {
        // Use the default signal masks
        sigset_t no_signals;
        sigset_t all_signals;
        sigemptyset (&no_signals);
        sigfillset (&all_signals);
        posix_spawnattr_setsigmask(&attr, &no_signals);
        posix_spawnattr_setsigdefault(&attr, &all_signals);

        err = posix_spawn (&pid,
                           progname,
                           file_actions,
                           &attr,
                           (char * const *)argv,
                           (char * const *)envp);
        
        posix_spawnattr_destroy(&attr);
    }
    return pid;
}

int 
main (int argc, char const **argv)
{
    char *buf = (char*) malloc (strlen (argv[0]) + 12);
    strlcpy (buf, argv[0], strlen (argv[0]) + 1);
    std::string directory_name (::dirname (buf));

    std::string other_program = directory_name + "/secondprog";
    int err = 0;    // Set breakpoint 1 here
    spawn_process (other_program.c_str(), argv, NULL, err);
    if (err)
        exit_with_errno (err, "posix_spawn x86_64 error");
    return 0;
}
