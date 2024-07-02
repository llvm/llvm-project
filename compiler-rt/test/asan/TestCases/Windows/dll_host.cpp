// UNSUPPORTED: target={{.*-windows-gnu}}

// This is a host program for DLL tests.
//
// Just make sure we can compile this.
// The actual compile&run sequence is to be done by the DLL tests.
// RUN: %clang_cl_asan -Od %s -Fe%t

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
