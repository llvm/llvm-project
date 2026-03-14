// Test to ensure that multiple rounds of dumping, using the
// __memprof_profile_reset interface to close the initial file
// and cause the profile to be reopened, works as expected.

// RUN: %clangxx_memprof  %s -o %t

// RUN: rm -f %t.log.*
// RUN: %env_memprof_opts=print_text=true:log_path=%t.log %run %t

// Check both outputs, starting with the renamed initial dump, then remove it so
// that the second glob matches a single file.
// RUN: FileCheck %s < %t.log.*.sv
// RUN: rm -f %t.log.*.sv
// RUN: FileCheck %s < %t.log.*
// CHECK: Memory allocation stack id

#include <sanitizer/memprof_interface.h>
#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <string>
int main(int argc, char **argv) {
  char *x = (char *)malloc(10);
  memset(x, 0, 10);
  free(x);
  __memprof_profile_dump();
  // Save the initial dump in a different file.
  std::string origname = __sanitizer_get_report_path();
  std::string svname = origname + ".sv";
  rename(origname.c_str(), svname.c_str());
  // This should cause the current file descriptor to be closed and the
  // the internal state reset so that the profile filename is reopened
  // on the next write.
  __memprof_profile_reset();
  // This will dump to origname again.
  __memprof_profile_dump();
  return 0;
}
