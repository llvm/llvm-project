#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

void handler(int sig)
{
    printf("Set a breakpoint here.\n");
    exit(0);
}

void abort_caller() {
    abort();
}

int main()
{
    if (signal(SIGABRT, handler) == SIG_ERR)
    {
        perror("signal");
        return 1;
    }

    abort_caller();
    return 2;
}
