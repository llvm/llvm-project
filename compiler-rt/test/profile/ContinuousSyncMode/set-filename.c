// RUN: %clang_pgogen -o %t.exe %s
// RUN: env LLVM_PROFILE_FILE="%c%t.profraw" %run %t.exe %t.profraw %t.bad

#include <string.h>

extern int __llvm_profile_is_continuous_mode_enabled(void);
extern void __llvm_profile_set_filename(const char *);
extern const char *__llvm_profile_get_filename();

int main(int argc, char **argv) {
  if (!__llvm_profile_is_continuous_mode_enabled())
    return 1;

  __llvm_profile_set_filename(argv[2]); // Try to set the filename to "%t.bad".
  const char *Filename = __llvm_profile_get_filename();
  return strcmp(Filename, argv[1]); // Check that the filename is "%t.profraw".
}
