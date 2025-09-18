// RUN: %clangxx -O2 %s -o %t

// Case 1: Try setting a path that is an invalid/inaccessible directory.
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=ERROR1
// RUN: %env_tool_opts=log_fallback_to_stderr=true %run %t 2>&1 | FileCheck %s --check-prefixes=ERROR1,FALLBACK

// Case 2: Try setting a path that is too large.
// RUN: not %run %t A 2>&1 | FileCheck %s --check-prefix=ERROR2
// RUN: %env_tool_opts=log_fallback_to_stderr=true %run %t A 2>&1 | FileCheck %s --check-prefixes=ERROR2,FALLBACK

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

int main(int argc, char **argv) {
  char buff[4096];
  if (argc == 1) {
    // Case 1
    sprintf(buff, "%s/report", argv[0]);
    // ERROR1: Can't create directory
  } else {
    // Case 2
    snprintf(buff, sizeof(buff), "%04095d", 42);
    // ERROR2: Path is too long
  }
  __sanitizer_set_report_path(buff);
}
// FALLBACK: falling back to stderr
