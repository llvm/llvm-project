// REQUIRES: darwin

// RUN: %clang_pgogen -o %t.exe %s
// RUN: env LLVM_PROFILE_FILE="%c%t.profraw" %run %t.exe %t.profraw
// RUN: env LLVM_PROFILE_FILE="%t%c.profraw" %run %t.exe %t.profraw
// RUN: env LLVM_PROFILE_FILE="%t.profraw%c" %run %t.exe %t.profraw

#include <string.h>
#include <stdio.h>

extern int __llvm_profile_is_continuous_mode_enabled(void);
extern const char *__llvm_profile_get_filename();
extern void __llvm_profile_set_dumped(void);

int main(int argc, char **argv) {
  if (!__llvm_profile_is_continuous_mode_enabled())
    return 1;

  // Check that the filename is "%t.profraw", followed by a null terminator.
  size_t n = strlen(argv[1]) + 1;
  const char *Filename = __llvm_profile_get_filename();

  for (int i = 0; i < n; ++i) {
    if (Filename[i] != argv[1][i]) {
      printf("Difference at: %d, Got: %c, Expected: %c\n", i, Filename[i], argv[1][i]);
      printf("Got: %s\n", Filename);
      printf("Expected: %s\n", argv[1]);
      return 1;
    }
  }
  return 0;
}
