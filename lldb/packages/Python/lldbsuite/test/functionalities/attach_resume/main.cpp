#include <stdio.h>
#include <fcntl.h>

#include <chrono>
#include <thread>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

volatile bool debugger_flag = true; // The debugger will flip this to false

void *start(void *data)
{
    int i;
    size_t idx = (size_t)data;
    for (i=0; i<30; i++)
    {
        if ( idx == 0 && debugger_flag)
            std::this_thread::sleep_for(std::chrono::microseconds(1)); // Set breakpoint here
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    return 0;
}

int main(int argc, char const *argv[])
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
    static_cast<void> (prctl_result);
#endif
#endif

    static const size_t nthreads = 16;
    std::thread threads[nthreads];
    size_t i;

    for (i=0; i<nthreads; i++)
        threads[i] = std::move(std::thread(start, (void*)i));

    for (i=0; i<nthreads; i++)
        threads[i].join();
}
