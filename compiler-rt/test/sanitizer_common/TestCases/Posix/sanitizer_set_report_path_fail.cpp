// RUN: %clangxx -O2 %s -o %t
// RUN: not %env %run %t 2>&1 | FileCheck %s --check-prefix=ERROR1
// RUN: not %env %run %t A 2>&1 | FileCheck %s --check-prefix=ERROR2

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

int main(int argc, char **argv) {
  char buff[4096];
  if (argc == 1) {
    // Try setting again with an invalid/inaccessible directory.
    sprintf(buff, "%s/report", argv[0]);
    // ERROR1: Can't create directory: {{.*}}
  } else {
    snprintf(buff, sizeof(buff), "%04095d", 42);
    // ERROR2: Path is too long: 00000000
  }
  __sanitizer_set_report_path(buff);
}
