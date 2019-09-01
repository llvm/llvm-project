#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/types.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/wait.h>

#if defined(PTRACE_ATTACH)
#define ATTACH_REQUEST PTRACE_ATTACH
#define DETACH_REQUEST PTRACE_DETACH
#elif defined(PT_ATTACH)
#define ATTACH_REQUEST PT_ATTACH
#define DETACH_REQUEST PT_DETACH
#else
#error "Unsupported platform"
#endif

bool writePid (const char* file_name, const pid_t pid)
{
    char *tmp_file_name = (char *)malloc(strlen(file_name) + 16);
    strcpy(tmp_file_name, file_name);
    strcat(tmp_file_name, "_tmp");
    int fd = open (tmp_file_name, O_CREAT | O_WRONLY | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd == -1)
    {
        fprintf (stderr, "open(%s) failed: %s\n", tmp_file_name, strerror (errno));
        free(tmp_file_name);
        return false;
    }
    char buffer[64];
    snprintf (buffer, sizeof(buffer), "%ld", (long)pid);

    bool res = true;
    if (write (fd, buffer, strlen (buffer)) == -1)
    {
        fprintf (stderr, "write(%s) failed: %s\n", buffer, strerror (errno));
        res = false;
    }
    close (fd);

    if (rename (tmp_file_name, file_name) == -1)
    {
        fprintf (stderr, "rename(%s, %s) failed: %s\n", tmp_file_name, file_name, strerror (errno));
        res = false;
    }
    free(tmp_file_name);

    return res;
}

void signal_handler (int)
{
}

int main (int argc, char const *argv[])
{
    if (argc < 2)
    {
        fprintf (stderr, "invalid number of command line arguments\n");
        return 1;
    }

    const pid_t pid = fork ();
    if (pid == -1)
    {
        fprintf (stderr, "fork failed: %s\n", strerror (errno));
        return 1;
    }

    if (pid > 0)
    {
        // Make pause call to return when a signal is received. Normally this happens when the
        // test runner tries to terminate us.
        signal (SIGHUP, signal_handler);
        signal (SIGTERM, signal_handler);
        if (ptrace (ATTACH_REQUEST, pid, NULL, 0) == -1)
        {
            fprintf (stderr, "ptrace(ATTACH) failed: %s\n", strerror (errno));
        }
        else
        {
            if (writePid (argv[1], pid))
                pause ();  // Waiting for the debugger trying attach to the child.

            if (ptrace (DETACH_REQUEST, pid, NULL, 0) != 0)
                fprintf (stderr, "ptrace(DETACH) failed: %s\n", strerror (errno));
        }

        kill (pid, SIGTERM);
        int status = 0;
        if (waitpid (pid, &status, 0) == -1)
            fprintf (stderr, "waitpid failed: %s\n", strerror (errno));
    }
    else
    {
        // child inferior.
        pause ();
    }

    printf ("Exiting now\n");
    return 0;
}
