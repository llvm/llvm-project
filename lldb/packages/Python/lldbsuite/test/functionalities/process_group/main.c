#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

volatile int release_child_flag = 0;

int main(int argc, char const *argv[])
{
    pid_t child = fork();
    if (child == -1)
    {
        perror("fork");
        return 1;
    }

    if (child > 0)
    { // parent
        if (argc < 2)
        {
            fprintf(stderr, "Need pid filename.\n");
            return 2;
        }

        // Let the test suite know the child's pid.
        FILE *pid_file = fopen(argv[1], "w");
        if (pid_file == NULL)
        {
            perror("fopen");
            return 3;
        }

        fprintf(pid_file, "%d\n", child);
        if (fclose(pid_file) == EOF)
        {
            perror("fclose");
            return 4;
        }

        // And wait for the child to finish it's work.
        int status = 0;
        pid_t wpid = wait(&status);
        if (wpid == -1)
        {
            perror("wait");
            return 5;
        }
        if (wpid != child)
        {
            fprintf(stderr, "wait() waited for wrong child\n");
            return 6;
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
        {
            fprintf(stderr, "child did not exit correctly\n");
            return 7;
        }
    }
    else
    { // child
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

        while (! release_child_flag) // Wait for debugger to attach
            sleep(1);

        printf("Child's previous process group is: %d\n", getpgid(0));
        setpgid(0, 0); // Set breakpoint here
        printf("Child's process group set to: %d\n", getpgid(0));
    }

    return 0;
}
