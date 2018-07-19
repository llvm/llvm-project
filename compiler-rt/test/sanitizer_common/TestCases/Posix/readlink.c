// RUN: %clang -O0 %s -o %t && %run %t

#include <assert.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv) {
  char symlink_path[PATH_MAX];
  snprintf(symlink_path, sizeof(symlink_path), "%s_%d.symlink", argv[0],
           getpid());
  remove(symlink_path);
  int res = symlink(argv[0], symlink_path);
  assert(!res);

  char readlink_path[PATH_MAX];
  ssize_t res2 = readlink(symlink_path, readlink_path, sizeof(readlink_path));
  assert(res2 >= 0);
  readlink_path[res2] = '\0';
  assert(!strcmp(readlink_path, argv[0]));

  return 0;
}
