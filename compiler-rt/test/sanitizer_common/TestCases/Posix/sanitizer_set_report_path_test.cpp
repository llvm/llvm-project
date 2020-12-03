// Test __sanitizer_set_report_path and __sanitizer_get_report_path:
// RUN: %clangxx -O2 %s -o %t
// RUN: %run %t | FileCheck %s

#include <assert.h>
#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <string.h>

volatile int *null = 0;

int main(int argc, char **argv) {
  char buff[1000];
  sprintf(buff, "%s.report_path", argv[0]);
  __sanitizer_set_report_path(buff);
  assert(strncmp(buff, __sanitizer_get_report_path(), strlen(buff)) == 0);
  printf("Path %s\n", __sanitizer_get_report_path());
}

// CHECK: Path {{.*}}Posix/Output/sanitizer_set_report_path_test.cpp.tmp.report_path.
