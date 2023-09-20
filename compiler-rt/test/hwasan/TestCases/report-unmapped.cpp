// RUN: %clangxx_hwasan %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <sanitizer/hwasan_interface.h>

#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char **argv) {
  const size_t kPS = sysconf(_SC_PAGESIZE);

  void *r = nullptr;
  int res = posix_memalign(&r, kPS, 2 * kPS);
  if (res) {
    perror("Failed to mmap: ");
    abort();
  }
  
  r = __hwasan_tag_pointer(r, 0);
  __hwasan_tag_memory(r, 1,  2 * kPS);
  
  // Disable access to the page just after tag-mismatch report address.
  res = mprotect((char*)r + kPS, kPS, PROT_NONE);
  if (res) {
    perror("Failed to mprotect: ");
    abort();
  }

  // Trigger tag-mismatch report.
  return *((char*)r + kPS - 1);
}

// CHECK: ERROR: HWAddressSanitizer: tag-mismatch on address
// CHECK: Memory tags around the buggy address
// CHECK: Tags for short granules around

// Check that report is complete.
// CHECK: SUMMARY: HWAddressSanitizer
