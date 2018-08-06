// RUN: %clangxx_scudo %s -o %t
// RUN: rm -rf %t-dir/random_shuffle_tmp_dir
// RUN: mkdir -p %t-dir/random_shuffle_tmp_dir
// RUN: %run %t 100 > %t-dir/random_shuffle_tmp_dir/out1
// RUN: %run %t 100 > %t-dir/random_shuffle_tmp_dir/out2
// RUN: %run %t 10000 > %t-dir/random_shuffle_tmp_dir/out1
// RUN: %run %t 10000 > %t-dir/random_shuffle_tmp_dir/out2
// RUN: not diff %t-dir/random_shuffle_tmp_dir/out?
// RUN: rm -rf %t-dir/random_shuffle_tmp_dir

// Tests that the allocator shuffles the chunks before returning to the user.

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
  int alloc_size = argc == 2 ? atoi(argv[1]) : 100;
  char *base = new char[alloc_size];
  for (int i = 0; i < 20; i++) {
    char *p = new char[alloc_size];
    printf("%zd\n", base - p);
  }
}
