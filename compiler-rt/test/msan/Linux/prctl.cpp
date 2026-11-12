// RUN: %clangxx_msan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <linux/prctl.h>
#include <sys/prctl.h>

int main(void) {
  prctl(PR_SET_NAME, "tname");
  char name[16];
  prctl(PR_GET_NAME, name);

  if (name[0] == 'A') {
    return 0;
  }
  if (name[5] != '\0') {
    return 0;
  }
  if (name[6] != '\0') {
    return 0;
  }
  // CHECK: SUMMARY: MemorySanitizer: use-of-uninitialized-value {{.*prctl.cpp}}:[[@LINE-3]]

  return 0;
}
