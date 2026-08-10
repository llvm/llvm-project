// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p 2>&1 | FileCheck %s
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %p 2>&1 | FileCheck %s
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %p 2>&1 | FileCheck %s

#include <assert.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include <sanitizer/msan_interface.h>

static void my_gl_closedir(void *dir) {
  if (!dir)
    exit(1);
  closedir((DIR *)dir);
}

static struct dirent *my_gl_readdir(void *dir) {
  if (!dir)
    exit(1);
  struct dirent *d = readdir((DIR *)dir);
  if (d) __msan_poison(d, d->d_reclen); // hehe
  return d;
}

static void *my_gl_opendir(const char *s) {
  assert(__msan_test_shadow(s, strlen(s) + 1) == (size_t)-1);
  return opendir(s);
}

static int my_gl_lstat(const char *s, struct stat *st) {
  assert(__msan_test_shadow(s, strlen(s) + 1) == (size_t)-1);
  if (!st)
    exit(1);
  return lstat(s, st);
}

static int my_gl_stat(const char *s, struct stat *st) {
  assert(__msan_test_shadow(s, strlen(s) + 1) == (size_t)-1);
  if (!st)
    exit(1);
  return lstat(s, st);
}

int main(int argc, char *argv[]) {
  assert(argc == 2);
  char buf[1024];
  snprintf(buf, sizeof(buf), "%s/%s", argv[1], "glob_test_root/*a");

  glob_t globbuf;
  globbuf.gl_closedir = my_gl_closedir;
  globbuf.gl_readdir = my_gl_readdir;
  globbuf.gl_opendir = my_gl_opendir;
  globbuf.gl_lstat = my_gl_lstat;
  globbuf.gl_stat = my_gl_stat;
  for (int i = 0; i < 10000; ++i) {
    int res = glob(buf, GLOB_ALTDIRFUNC | GLOB_MARK, 0, &globbuf);
    assert(res == 0);
    printf("%d %s\n", errno, strerror(errno));
    assert(globbuf.gl_pathc == 2);
    printf("%zu\n", strlen(globbuf.gl_pathv[0]));
    printf("%zu\n", strlen(globbuf.gl_pathv[1]));
    __msan_poison(globbuf.gl_pathv[0], strlen(globbuf.gl_pathv[0]) + 1);
    __msan_poison(globbuf.gl_pathv[1], strlen(globbuf.gl_pathv[1]) + 1);
    globfree(&globbuf);
  }

  printf("PASS\n");
  // CHECK: PASS
  return 0;
}
