// UNSUPPORTED: target={{.*-windows-gnu}}

// This is a host program for DLL tests.
//
// Just make sure we can compile this.
// The actual compile&run sequence is to be done by the DLL tests.
// RUN: %clang_cl_asan -Od %s -Fe%t
//
// Get the list of ASan wrappers exported by the main module RTL:
// note: The mangling decoration (i.e. @4 )is removed because calling convention
//       differ from 32-bit and 64-bit.
// RUN: dumpbin /EXPORTS %t | grep -o "__asan_wrap[^ ]*" | sed -e s/@.*// > %t.exports1
//
// The exception handlers differ in 32-bit and 64-bit, so we ignore them:
// RUN: grep '[E]XPORT:' %s | sed -e 's/.*[E]XPORT: //' > %t.exports2
// EXPORT: __asan_wrap__except_handler3
// EXPORT: __asan_wrap__except_handler4
// EXPORT: __asan_wrap___C_specific_handler
//
// Get the list of ASan wrappers imported by the DLL RTL:
// [BEWARE: be really careful with the sed commands, as this test can be run
//  from different environments with different shells and seds]
// RUN: grep INTERCEPT_LIBRARY_FUNCTION %p/../../../../lib/asan/asan_win_dll_thunk.cpp \
// RUN:  | grep -v define | sed -e s/.*(/__asan_wrap_/ -e s/).*//              \
// RUN:  > %t.imports1
//
// Add functions interecepted in asan_malloc.win.cpp and asan_win.cpp.
// RUN: grep '[I]MPORT:' %s | sed -e 's/.*[I]MPORT: //' > %t.imports2
// IMPORT: __asan_wrap_HeapAlloc
// IMPORT: __asan_wrap_HeapFree
// IMPORT: __asan_wrap_HeapReAlloc
// IMPORT: __asan_wrap_HeapSize
// IMPORT: __asan_wrap_CreateThread
// IMPORT: __asan_wrap_RaiseException
// IMPORT: __asan_wrap_RtlRaiseException
// IMPORT: __asan_wrap_SetUnhandledExceptionFilter
// IMPORT: __asan_wrap_RtlSizeHeap
// IMPORT: __asan_wrap_RtlAllocateHeap
// IMPORT: __asan_wrap_RtlReAllocateHeap
// IMPORT: __asan_wrap_RtlFreeHeap
//
// RUN: cat %t.imports1 %t.imports2 | sort | uniq > %t.imports-sorted
// RUN: cat %t.exports1 %t.exports2 | sort | uniq > %t.exports-sorted
//
// Now make sure the DLL thunk imports everything:
// RUN: echo
// RUN: echo "=== NOTE === If you see a mismatch below, please update asan_win_dll_thunk.cpp"
// RUN: diff %t.imports-sorted %t.exports-sorted
// REQUIRES: asan-static-runtime

#include <stdio.h>
#include <windows.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s [client].dll\n", argv[0]);
    return 101;
  }

  const char *dll_name = argv[1];

  HMODULE h = LoadLibrary(dll_name);
  if (!h) {
    DWORD err = GetLastError();
    printf("Could not load DLL: %s (code: %lu)!\n", dll_name, err);

    LPSTR buf;

    FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                       FORMAT_MESSAGE_IGNORE_INSERTS,
                   NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), buf, 0,
                   NULL);

    printf("Error: %s\n", buf);

    LocalFree(buf);

    return 102;
  }

  typedef int (*test_function)();
  test_function gf = (test_function)GetProcAddress(h, "test_function");
  if (!gf) {
    printf("Could not locate test_function in the DLL!\n");
    FreeLibrary(h);
    return 103;
  }

  int ret = gf();

  FreeLibrary(h);
  return ret;
}
