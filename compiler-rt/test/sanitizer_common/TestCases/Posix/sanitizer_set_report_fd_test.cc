// Test __sanitizer_set_report_fd:
// RUN: %clangxx -O2 %s -o %t
// RUN: not %run %t 2>&1   | FileCheck %s
// RUN: not %run %t stdout | FileCheck %s
// RUN: not %run %t %t-out && FileCheck < %t-out %s

// REQUIRES: stable-runtime
// XFAIL: android && asan

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>

volatile int *null = 0;

int main(int argc, char **argv) {
  if (argc == 2) {
    if (!strcmp(argv[1], "stdout")) {
      __sanitizer_set_report_fd(reinterpret_cast<void*>(1));
    } else {
      int fd = open(argv[1], O_CREAT | O_WRONLY | O_TRUNC, S_IRWXU);
      assert(fd > 0);
      __sanitizer_set_report_fd(reinterpret_cast<void*>(fd));
    }
  }
  *null = 0;
}

// CHECK: ERROR: {{.*}} SEGV on unknown address
