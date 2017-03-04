// This header is included in all the test programs (C and C++) and provides a
// hook for dealing with platform-specifics.
#if defined(_WIN32) || defined(_WIN64)
#ifdef COMPILING_LLDB_TEST_DLL
#define LLDB_TEST_API __declspec(dllexport)
#else
#define LLDB_TEST_API __declspec(dllimport)
#endif
#else
#define LLDB_TEST_API
#endif

#if defined(_WIN32)
#define LLVM_PRETTY_FUNCTION __FUNCSIG__
#else
#define LLVM_PRETTY_FUNCTION LLVM_PRETTY_FUNCTION
#endif


// On some systems (e.g., some versions of linux) it is not possible to attach to a process
// without it giving us special permissions. This defines the lldb_enable_attach macro, which
// should perform any such actions, if needed by the platform. This is a macro instead of a
// function to avoid the need for complex linking of the test programs.
#if defined(__linux__)
#include <sys/prctl.h>

// Android API <= 16 does not have these defined.
#ifndef PR_SET_PTRACER
#define PR_SET_PTRACER 0x59616d61
#endif
#ifndef PR_SET_PTRACER_ANY
#define PR_SET_PTRACER_ANY ((unsigned long)-1)
#endif

// For now we execute on best effort basis.  If this fails for some reason, so be it.
#define lldb_enable_attach()                                                          \
    do                                                                                \
    {                                                                                 \
        const int prctl_result = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);  \
        (void)prctl_result;                                                           \
    } while (0)

#else // not linux

#define lldb_enable_attach()

#endif

#if defined(__APPLE__) && defined(LLDB_USING_LIBSTDCPP)              

// on Darwin, libstdc++ is missing <atomic>, so this would cause any test to fail building
// since this header file is being included in every C-family test case, we need to not include it
// on Darwin, most tests use libc++ by default, so this will only affect tests that explicitly require libstdc++

#else
#ifdef __cplusplus
#include <atomic>

// Note that although hogging the CPU while waiting for a variable to change
// would be terrible in production code, it's great for testing since it
// avoids a lot of messy context switching to get multiple threads synchronized.

typedef std::atomic<int> pseudo_barrier_t;
#define pseudo_barrier_wait(barrier)        \
    do                                      \
    {                                       \
        --(barrier);                        \
        while ((barrier).load() > 0)        \
            ;                               \
    } while (0)

#define pseudo_barrier_init(barrier, count) \
    do                                      \
    {                                       \
        (barrier) = (count);                \
    } while (0)
#endif // __cplusplus
#endif // defined(__APPLE__) && defined(LLDB_USING_LIBSTDCPP)
