// UNSUPPORTED: windows
// RUN: %clang_pgogen -o %t.bin %s -DTESTPATH=\"%t.dir\"
// RUN: rm -rf %t.dir
// RUN: %run %t.bin

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

void __llvm_profile_set_dir_mode(unsigned Mode);
unsigned __llvm_profile_get_dir_mode(void);
void __llvm_profile_recursive_mkdir(char *Path);

static int test(unsigned Mode, const char *TestDir) {
  int Ret = 0;

  /* Create a dir and set the mode accordingly. */
  char *Dir = strdup(TestDir);
  if (!Dir)
    return -1;
  __llvm_profile_set_dir_mode(Mode);
  __llvm_profile_recursive_mkdir(Dir);

  if (Mode != __llvm_profile_get_dir_mode())
    Ret = -1;
  else {
    const unsigned Expected = ~umask(0) & Mode;
    struct stat DirSt;
    if (stat(Dir, &DirSt) == -1)
      Ret = -1;
    else if (DirSt.st_mode != Expected) {
      printf("Modes do not match: Expected %o but found %o (%s)\n", Expected,
             DirSt.st_mode, Dir);
      Ret = -1;
    }
  }

  free(Dir);
  return Ret;
}

int main(void) {
  if (test(S_IFDIR | 0777, TESTPATH "/foo/bar/baz/") ||
      test(S_IFDIR | 0666, TESTPATH "/foo/bar/qux/"))
    return -1;
  return 0;
}
