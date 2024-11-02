// XFAIL: aix
// Test __llvm_profile_get_filename.
// RUN: %clang_pgogen -O2 -o %t %s
// RUN: %run %t

#include <stdio.h>
#include <string.h>

const char *__llvm_profile_get_filename();
void __llvm_profile_set_filename(const char *);

int main(int argc, const char *argv[]) {
  int i;
  const char *filename;
  const char *new_filename = "/path/to/test.profraw";

  filename = __llvm_profile_get_filename();
  if (strncmp(filename, "default_", 8)) {
    fprintf(stderr,
            "Error: got filename %s, expected it to start with 'default_'\n",
            filename);
    return 1;
  }
  if (strcmp(filename + strlen(filename) - strlen(".profraw"), ".profraw")) {
    fprintf(stderr,
            "Error: got filename %s, expected it to end with '.profraw'\n",
            filename);
    return 1;
  }

  __llvm_profile_set_filename(new_filename);
  filename = __llvm_profile_get_filename();
  if (strcmp(filename, new_filename)) {
    fprintf(stderr, "Error: got filename %s, expected '%s'\n", filename,
            new_filename);
    return 1;
  }

  return 0;
}
