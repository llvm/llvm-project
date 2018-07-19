// RUN: %clang -O0 %s -o %t && %run %t

#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  char symlink_path[PATH_MAX];
  snprintf(symlink_path, sizeof(symlink_path), "%s_%d.symlink", argv[0],
           getpid());
  remove(symlink_path);
  int res = symlink(argv[0], symlink_path);
  assert(!res);

  char readlinkat_path[PATH_MAX];
  int res2 = readlinkat(AT_FDCWD, symlink_path, readlinkat_path,
                        sizeof(readlinkat_path));
  assert(res2 >= 0);
  readlinkat_path[res2] = '\0';
  assert(!strcmp(readlinkat_path, argv[0]));

  return 0;
}
