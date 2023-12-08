// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android

// RUN: rm -rf %t-dir
// RUN: mkdir -p %t-dir
//
// RUN: %clangxx_asan -O0 %s -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s
//
// RUN: %clangxx_asan -O0 %s -D_FILE_OFFSET_BITS=64 -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 %s -D_FILE_OFFSET_BITS=64 -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 %s -D_FILE_OFFSET_BITS=64 -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 %s -D_FILE_OFFSET_BITS=64 -DTEMP_DIR='"'"%t-dir"'"' -o %t && %run %t 2>&1 | FileCheck %s

#include <dirent.h>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


int main() {
  // Ensure the readdir_r interceptor doesn't erroneously mark the entire dirent
  // as written when the end of the directory pointer is reached.
  fputs("test1: reading the " TEMP_DIR " directory...\n", stderr);
  DIR *d = opendir(TEMP_DIR);
  struct dirent *result = (struct dirent *)(0xfeedbeef);
  // We assume the temp dir for this test doesn't have crazy long file names.
  char entry_buffer[4096];
  memset(entry_buffer, 0xab, sizeof(entry_buffer));
  unsigned count = 0;
  do {
    // Stamp the entry struct to try to trick the interceptor.
    ((struct dirent *)entry_buffer)->d_reclen = 9999;
    if (readdir_r(d, (struct dirent *)entry_buffer, &result) != 0)
      abort();
    ++count;
  } while (result != NULL);
  fprintf(stderr, "read %d entries\n", count);
  closedir(d);
  // CHECK: test1: reading the {{.*}} directory...
  // CHECK-NOT: stack-buffer-overflow
  // CHECK: read {{.*}} entries

  // Ensure the readdir64_r interceptor doesn't have the bug either.
  fputs("test2: reading the " TEMP_DIR " directory...\n", stderr);
  d = opendir(TEMP_DIR);
  struct dirent64 *result64;
  memset(entry_buffer, 0xab, sizeof(entry_buffer));
  count = 0;
  do {
    // Stamp the entry struct to try to trick the interceptor.
    ((struct dirent64 *)entry_buffer)->d_reclen = 9999;
    if (readdir64_r(d, (struct dirent64 *)entry_buffer, &result64) != 0)
      abort();
    ++count;
  } while (result64 != NULL);
  fprintf(stderr, "read %d entries\n", count);
  closedir(d);
  // CHECK: test2: reading the {{.*}} directory...
  // CHECK-NOT: stack-buffer-overflow
  // CHECK: read {{.*}} entries
}
