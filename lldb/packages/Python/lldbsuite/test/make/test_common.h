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

#if defined(__cplusplus) && defined(_MSC_VER) && (_HAS_EXCEPTIONS == 0)
// Compiling MSVC libraries with _HAS_EXCEPTIONS=0, eliminates most but not all
// calls to __uncaught_exception.  Unfortunately, it does seem to eliminate
// the delcaration of __uncaught_excpeiton.  Including <eh.h> ensures that it is
// declared.  This may not be necessary after MSVC 12.
#include <eh.h>
#endif


// On some systems (e.g., some versions of linux) it is not possible to attach to a process
// without it giving us special permissions. This defines the lldb_enable_attach macro, which
// should perform any such actions, if needed by the platform. This is a macro instead of a
// function to avoid the need for complex linking of the test programs.
#if defined(__linux__)
#include <sys/prctl.h>

#if defined(PR_SET_PTRACER) && defined(PR_SET_PTRACER_ANY)
// For now we execute on best effort basis.  If this fails for some reason, so be it.
#define lldb_enable_attach()                                                          \
    do                                                                                \
    {                                                                                 \
        const int prctl_result = prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);  \
        (void)prctl_result;                                                           \
    } while (0)

#endif

#else // not linux

#define lldb_enable_attach()

#endif
