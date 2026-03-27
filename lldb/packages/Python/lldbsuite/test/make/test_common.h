// This header is included in all the test programs (C and C++) and provides a
// hook for dealing with platform-specifics.

#if defined(_WIN32) || defined(_WIN64)
#define LLDB_DYLIB_EXPORT __declspec(dllexport)
#define LLDB_DYLIB_IMPORT __declspec(dllimport)
#else
#define LLDB_DYLIB_EXPORT
#define LLDB_DYLIB_IMPORT
#endif

#ifdef COMPILING_LLDB_TEST_DLL
#define LLDB_TEST_API LLDB_DYLIB_EXPORT
#else
#define LLDB_TEST_API LLDB_DYLIB_IMPORT
#endif

#if defined(_WIN32)
#define LLVM_PRETTY_FUNCTION __FUNCSIG__
#else
#define LLVM_PRETTY_FUNCTION LLVM_PRETTY_FUNCTION
#endif
