// Test __llvm_profile_get_filename when the on-line merging mode is enabled.
//
// RUN: %clang_pgogen -fPIC -shared -o %t.dso %p/../Inputs/instrprof-get-filename-dso.c
// RUN: %clang_pgogen -o %t %s %t.dso
// RUN: env LLVM_PROFILE_FILE="%t-%m.profraw" %run %t

#include <string.h>

const char *__llvm_profile_get_filename(void);
extern const char *get_filename_from_DSO(void);

int main(int argc, const char *argv[]) {
  const char *filename1 = __llvm_profile_get_filename();
  const char *filename2 = get_filename_from_DSO();

  // Exit with code 1 if the two filenames are the same.
  return strcmp(filename1, filename2) == 0;
}
