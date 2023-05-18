// RUN: %clangxx_msan -O0 %s -o %t && %run %t %p
// RUN: %clangxx_msan -O0 -D_FILE_OFFSET_BITS=64 %s -o %t && %run %t %p
// RUN: %clangxx_msan -O3 %s -o %t && %run %t %p

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


static int my_filter(const struct dirent *a) {
  assert(__msan_test_shadow(&a, sizeof(a)) == (size_t)-1);
  printf("%s\n", a->d_name);
  __msan_print_shadow(a, a->d_reclen);
  assert(__msan_test_shadow(a, a->d_reclen) == (size_t)-1);
  printf("%s\n", a->d_name);
  return strlen(a->d_name) == 3 && a->d_name[2] == 'b';
}

static int my_compar(const struct dirent **a, const struct dirent **b) {
  assert(__msan_test_shadow(a, sizeof(*a)) == (size_t)-1);
  assert(__msan_test_shadow(*a, (*a)->d_reclen) == (size_t)-1);
  assert(__msan_test_shadow(b, sizeof(*b)) == (size_t)-1);
  assert(__msan_test_shadow(*b, (*b)->d_reclen) == (size_t)-1);
  if ((*a)->d_name[1] == (*b)->d_name[1])
    return 0;
  return ((*a)->d_name[1] < (*b)->d_name[1]) ? 1 : -1;
}

int main(int argc, char *argv[]) {
  assert(argc == 2);
  char buf[1024];
  snprintf(buf, sizeof(buf), "%s/%s", argv[1], "scandir_test_root/");
  
  struct dirent **d;
  int res = scandir(buf, &d, my_filter, my_compar);
  assert(res == 2);
  assert(__msan_test_shadow(&d, sizeof(*d)) == (size_t)-1);
  for (int i = 0; i < res; ++i) {
    assert(__msan_test_shadow(&d[i], sizeof(d[i])) == (size_t)-1);
    assert(__msan_test_shadow(d[i], d[i]->d_reclen) == (size_t)-1);
  }

  assert(strcmp(d[0]->d_name, "bbb") == 0);
  assert(strcmp(d[1]->d_name, "aab") == 0);
  return 0;
}
