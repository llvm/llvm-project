#include <stdio.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <chrono>
#include <thread>

int main(int argc, char const *argv[]) {
    int temp;
#if defined(__linux__)
    // Immediately enable any ptracer so that we can allow the stub attach
    // operation to succeed.  Some Linux kernels are locked down so that
    // only an ancestor process can be a ptracer of a process.  This disables that
    // restriction.  Without it, attach-related stub tests will fail.
#if defined(PR_SET_PTRACER) && defined(PR_SET_PTRACER_ANY)
    int prctl_result;

    // For now we execute on best effort basis.  If this fails for
    // some reason, so be it.
    prctl_result = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    (void) prctl_result;
#endif
#endif

    // Waiting to be attached by the debugger.
    temp = 0;

    while (temp < 30) // Waiting to be attached...
    {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        temp++;
    }

    printf("Exiting now\n");
}
