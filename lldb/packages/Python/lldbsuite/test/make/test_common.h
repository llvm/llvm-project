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
