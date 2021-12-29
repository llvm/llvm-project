// UNSUPPORTED: windows
// The sanitizer-windows bot is saying:
// instrprof-write-buffer-internal.c.tmp.buf.profraw: Invalid instrumentation profile data (file header is corrupt)

// RUN: rm -f %t.buf.profraw %t.profraw
// RUN: %clang_profgen -w -o %t %s
// RUN: env LLVM_PROFILE_FILE=%t.profraw %run %t %t.buf.profraw
// RUN: llvm-profdata show %t.buf.profraw | FileCheck %s -check-prefix=WRITE-BUFFER
// RUN: not llvm-profdata show %t.profraw 2>&1 | FileCheck %s -check-prefix=ALREADY-DUMPED

// WRITE-BUFFER: Instrumentation level: Front-end
// WRITE-BUFFER: Total functions: 1
// WRITE-BUFFER: Maximum function count: 1
// WRITE-BUFFER: Maximum internal block count: 0

// ALREADY-DUMPED: error: {{.+}}: empty raw profile file

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

const void *__llvm_profile_begin_data(void);
const void *__llvm_profile_end_data(void);
const char *__llvm_profile_begin_names(void);
const char *__llvm_profile_end_names(void);
char *__llvm_profile_begin_counters(void);
char *__llvm_profile_end_counters(void);

uint64_t __llvm_profile_get_size_for_buffer_internal(
    const void *DataBegin, const void *DataEnd, const char *CountersBegin,
    const char *CountersEnd, const char *NamesBegin, const char *NamesEnd);

int __llvm_profile_write_buffer_internal(char *Buffer, const void *DataBegin,
                                         const void *DataEnd,
                                         const char *CountersBegin,
                                         const char *CountersEnd,
                                         const char *NamesBegin,
                                         const char *NamesEnd);

void __llvm_profile_set_dumped(void);

int main(int argc, const char *argv[]) {
  uint64_t bufsize = __llvm_profile_get_size_for_buffer_internal(
      __llvm_profile_begin_data(), __llvm_profile_end_data(),
      __llvm_profile_begin_counters(), __llvm_profile_end_counters(),
      __llvm_profile_begin_names(), __llvm_profile_end_names());

  char *buf = malloc(bufsize);
  int ret = __llvm_profile_write_buffer_internal(buf,
      __llvm_profile_begin_data(), __llvm_profile_end_data(),
      __llvm_profile_begin_counters(), __llvm_profile_end_counters(),
      __llvm_profile_begin_names(), __llvm_profile_end_names());

  if (ret != 0) {
    fprintf(stderr, "failed to write buffer");
    return ret;
  }

  FILE *f = fopen(argv[1], "w");
  fwrite(buf, bufsize, 1, f);
  fclose(f);
  free(buf);

  __llvm_profile_set_dumped();

  return 0;
}
