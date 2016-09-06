// Test the handle_sigill option.
//
// RUN: %clangxx_asan %s -o %t && %env_asan_opts=handle_sigill=0 not --crash %run %t 2>&1 | FileCheck %s --check-prefix=CHECK0
// RUN: %clangxx_asan %s -o %t && %env_asan_opts=handle_sigill=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK1
// REQUIRES: x86-target-arch
// UNSUPPORTED: darwin

#ifdef _WIN32
#include <windows.h>
#endif

int main(int argc, char **argv) {
#ifdef _WIN32
  // Sometimes on Windows this test generates a WER fault dialog. Suppress that.
  UINT new_flags = SEM_FAILCRITICALERRORS |
                   SEM_NOGPFAULTERRORBOX |
                   SEM_NOOPENFILEERRORBOX;
  // Preserve existing error mode, as discussed at
  // http://blogs.msdn.com/oldnewthing/archive/2004/07/27/198410.aspx
  UINT existing_flags = SetErrorMode(new_flags);
  SetErrorMode(existing_flags | new_flags);
#endif

  if (argc)
    __builtin_trap();
  // Unreachable code to avoid confusing the Windows unwinder.
#ifdef _WIN32
  SetErrorMode(0);
#endif
}
// CHECK0-NOT: ERROR: AddressSanitizer
// CHECK1: ERROR: AddressSanitizer: {{ILL|illegal-instruction}} on unknown address {{0x0*}}
