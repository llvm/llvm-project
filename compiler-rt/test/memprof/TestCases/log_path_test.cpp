// The for loop in the backticks below requires bash.
// REQUIRES: shell
//
// RUN: %clangxx_memprof  %s -o %t

// Regular run.
// RUN: %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-GOOD < %t.out

// Good log_path.
// RUN: rm -f %t.log.*
// RUN: %env_memprof_opts=log_path=%t.log %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-GOOD < %t.log.*

// Invalid log_path.
// RUN: %env_memprof_opts=log_path=/dev/null/INVALID not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-INVALID < %t.out

// Too long log_path.
// RUN: %env_memprof_opts=log_path=`for((i=0;i<10000;i++)); do echo -n $i; done` \
// RUN:   not %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-LONG < %t.out

// Specifying the log name via the __memprof_profile_filename variable.
// RUN: %clangxx_memprof  %s -o %t -DPROFILE_NAME_VAR="%t.log2"
// RUN: rm -f %t.log2.*
// RUN: %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-GOOD < %t.log2.*

// Check that the log_path option overrides the log name set via the
// __memprof_profile_filename variable.
// RUN: rm -f %t.log.*
// RUN: %env_memprof_opts=log_path=%t.log %run %t 2> %t.out
// RUN: FileCheck %s --check-prefix=CHECK-GOOD < %t.log.*

#ifdef PROFILE_NAME_VAR
#define xstr(s) str(s)
#define str(s) #s
char __memprof_profile_filename[] = xstr(PROFILE_NAME_VAR);
#endif

#include <stdlib.h>
#include <string.h>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  return 0;
}
// CHECK-GOOD: Memory allocation stack id
// CHECK-INVALID: ERROR: Can't open file: /dev/null/INVALID
// CHECK-LONG: ERROR: Path is too long: 01234
