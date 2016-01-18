//===-- main.cpp ------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include <stdio.h>

#if defined(__linux__)
#include <sys/prctl.h>
#endif

#include <chrono>
#include <thread>

long double outermost_return_long_double (long double my_long_double);

int main (int argc, char const *argv[])
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

    char my_string[] = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 0};
    double my_double = 1234.5678;
    long double my_long_double = 1234.5678;

    // For simplicity assume that any cmdline argument means wait for attach.
    if (argc > 1)
    {
        volatile int wait_for_attach=1;
        while (wait_for_attach)
            std::this_thread::sleep_for(std::chrono::microseconds(1));
    }

    printf("my_string=%s\n", my_string);
    printf("my_double=%g\n", my_double);
    outermost_return_long_double (my_long_double);
    return 0;
}
