// RUN: %clang_pgogen -O2 -o %t.0 %s
// RUN: %clang_pgogen=%t.d1 -O2 -o %t.1 %s
// RUN: %clang_pgogen=%t.d1/%t.d2 -O2 -o %t.2 %s
// RUN: %clang_pgogen=a/b/c/d -O2 -o %t.3 %s
// RUN: %clang_pgogen=/a/b/c/d -O2 -o %t.4 %s
//
// RUN: %run %t.0  ""
// RUN: env LLVM_PROFILE_FILE=%t.d1/default.profraw %run %t.0  %t.d1/
// RUN: env LLVM_PROFILE_FILE=%t.d1/%t.d2/default.profraw %run %t.0 %t.d1/%t.d2/
// RUN: %run %t.1 %t.d1/
// RUN: %run %t.2 %t.d1/%t.d2/
// RUN: %run %t.2 %t.d1/%t.d2/ %t.d1/%t.d2/%t.d3/blah.profraw %t.d1/%t.d2/%t.d3/

// RUN: env GCOV_PREFIX=%t.prefix %run %t.3  %t.prefix/a/b/c/d/
// RUN: env GCOV_PREFIX=%t.prefix env GCOV_PREFIX_STRIP=1 %run %t.3 %t.prefix/b/c/d/ 
// RUN: env GCOV_PREFIX=%t.prefix env GCOV_PREFIX_STRIP=2 %run  %t.3  %t.prefix/c/d/ 
// RUN: env GCOV_PREFIX=%t.prefix env GCOV_PREFIX_STRIP=3 %run %t.3  %t.prefix/d/

// RUN: env GCOV_PREFIX=%t.prefix %run %t.4  %t.prefix/a/b/c/d/
// RUN: env GCOV_PREFIX=%t.prefix env GCOV_PREFIX_STRIP=1 %run %t.4 %t.prefix/b/c/d/ 
// RUN: env GCOV_PREFIX=%t.prefix env GCOV_PREFIX_STRIP=2 %run %t.4  %t.prefix/c/d/ 
// RUN: env GCOV_PREFIX=%t.prefix env GCOV_PREFIX_STRIP=3 %run %t.4  %t.prefix/d/

#include <stdio.h>
#include <string.h>

const char *__llvm_profile_get_path_prefix();
void __llvm_profile_set_filanem(const char*);

int main(int argc, const char *argv[]) {
  int i;
  const char *expected;
  const char *prefix;
  if (argc < 2)
    return 1;

  expected = argv[1];
  prefix = __llvm_profile_get_path_prefix();

  if (strcmp(prefix, expected)) {
    fprintf(stderr, "Expected = %s\n", expected);
    fprintf(stderr, "  Actual = %s\n", prefix);
    return 1;
  }

  if (argc == 4) {
    __llvm_profile_set_filename(argv[2]);
    prefix = __llvm_profile_get_path_prefix();
    expected = argv[3];
    if (strcmp(prefix, expected))
      return 1;
  }

  return 0;
}
