// RUN: %clang_pgogen -o %t.exe %s
// RUN: env LLVM_PROFILE_FILE="%c%t.profraw" %run %t.exe %t.bad 2>&1 | FileCheck %s

// CHECK: __llvm_profile_set_file_object(fd={{[0-9]+}}) not supported
// CHECK: Profile data not written to file: already written.

#include <stdio.h>

extern int __llvm_profile_is_continuous_mode_enabled(void);
extern void __llvm_profile_set_file_object(FILE *, int);
extern int __llvm_profile_write_file(void);

int main(int argc, char **argv) {
  if (!__llvm_profile_is_continuous_mode_enabled())
    return 1;

  FILE *f = fopen(argv[1], "a+b");
  if (!f)
    return 1;

  __llvm_profile_set_file_object(f, 0); // Try to set the file to "%t.bad".

  if (__llvm_profile_write_file() != 0)
    return 1;

  f = fopen(argv[1], "r");
  if (!f)
    return 1;

  fseek(f, 0, SEEK_END);
  return ftell(f); // Check that the "%t.bad" is empty.
}
