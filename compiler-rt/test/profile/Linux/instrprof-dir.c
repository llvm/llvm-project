// RUN: %clang_pgogen -o %t %s
// RUN: env LLVM_PROFILE_FILE="%t.d/%m.profraw"
// RUN: rm -fr %t.d
// RUN: %run %t %t.d

#include <errno.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (access(argv[1], F_OK) == 0)
    return 1; // %t.d should not exist yet.
  return !(errno == ENOENT);
}
