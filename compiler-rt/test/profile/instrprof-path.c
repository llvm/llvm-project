// XFAIL: aix
// RUN: %clang_pgogen -O2 -o %t.0 %s
// RUN: %clang_pgogen=%/t.d1 -O2 -o %t.1 %s
// RUN: %clang_pgogen=%/t.d1/%:t.d2 -O2 -o %t.2 %s
//
// RUN: %run %t.0  ""
// RUN: env LLVM_PROFILE_FILE=%/t.d1/default.profraw %run %t.0  %/t.d1
// RUN: env LLVM_PROFILE_FILE=%/t.d1/%:t.d2/default.profraw %run %t.0 %/t.d1/%:t.d2
// RUN: %run %t.1 %/t.d1
// RUN: %run %t.2 %/t.d1/%:t.d2
// RUN: %run %t.2 %/t.d1/%:t.d2 %/t.d1/%:t.d2/%:t.d3/blah.profraw %/t.d1/%:t.d2/%:t.d3/

#include <string.h>

const char *__llvm_profile_get_path_prefix();
void __llvm_profile_set_filename(const char*);

int main(int argc, const char *argv[]) {
  const char *expected;
  const char *prefix;
  if (argc < 2)
    return 1;

  expected = argv[1];
  prefix = __llvm_profile_get_path_prefix();

  // The last character should be a trailing slash. Ignore it in the comparison
  // since it could be '/' or '\\'.
  int slashpos = strlen(prefix);
  if (slashpos > 0) {
    --slashpos;
    if (prefix[slashpos] != '/' && prefix[slashpos] != '\\')
      return 1;
  }

  if (strncmp(prefix, expected, slashpos))
    return 1;

  if (argc == 4) {
    __llvm_profile_set_filename(argv[2]);
    prefix = __llvm_profile_get_path_prefix();
    expected = argv[3];
    if (strcmp(prefix, expected))
      return 1;
  }

  return 0;
}
