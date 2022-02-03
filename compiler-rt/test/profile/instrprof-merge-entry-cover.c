// RUN: %clang_pgogen -O2 -mllvm -pgo-function-entry-coverage -o %t %s
// RUN: %run %t %t.profraw 1 1
// RUN: llvm-profdata show --all-functions --counts %t.profraw  | FileCheck %s

// FIXME: llvm-profdata exits with "Malformed instrumentation profile data"
// XFAIL: msvc

#include "profile_test.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int __llvm_profile_runtime = 0;
uint64_t __llvm_profile_get_size_for_buffer(void);
int __llvm_profile_write_buffer(char *);
void __llvm_profile_reset_counters(void);
int __llvm_profile_merge_from_buffer(const char *, uint64_t);

__attribute__((noinline)) int dumpBuffer(const char *FileN, const char *Buffer,
                                         uint64_t Size) {
  FILE *File = fopen(FileN, "w");
  if (!File)
    return 1;
  if (fwrite(Buffer, 1, Size, File) != Size)
    return 1;
  return fclose(File);
}

int g = 0;
__attribute__((noinline)) void foo(char c) {
  if (c == '1')
    g++;
  else
    g--;
}

/* This function is not profiled */
__attribute__((noinline)) void bar(int M) { g += M; }

int main(int argc, const char *argv[]) {
  int i;
  if (argc < 4)
    return 1;

  const uint64_t MaxSize = 10000;
  static ALIGNED(sizeof(uint64_t)) char Buffer[MaxSize];

  uint64_t Size = __llvm_profile_get_size_for_buffer();
  if (Size > MaxSize)
    return 1;

  /* Start profiling. */
  __llvm_profile_reset_counters();
  foo(argv[2][0]);
  /* End profiling by freezing counters. */
  if (__llvm_profile_write_buffer(Buffer))
    return 1;

  /* Its profile will be discarded. */
  for (i = 0; i < 10; i++)
    bar(1);

  /* Start profiling again and merge in previously
     saved counters in buffer. */
  __llvm_profile_reset_counters();
  __llvm_profile_merge_from_buffer(Buffer, Size);
  foo(argv[3][0]);
  /* End profiling */
  if (__llvm_profile_write_buffer(Buffer))
    return 1;

  /* Its profile will be discarded. */
  bar(2);

  /* Now it is time to dump the profile to file.  */
  return dumpBuffer(argv[1], Buffer, Size);
}

// CHECK-LABEL: dumpBuffer:
// CHECK:        Counters: 1
// CHECK-NEXT:    Block counts: [0]

// CHECK-LABEL:  foo:
// CHECK:         Counters: 1
// CHECK-NEXT:    Block counts: [1]

// CHECK-LABEL:  bar:
// CHECK:         Counters: 1
// CHECK-NEXT:    Block counts: [0]

// CHECK-LABEL:  main:
// CHECK:         Counters: 1
// CHECK-NEXT:    Block counts: [0]
