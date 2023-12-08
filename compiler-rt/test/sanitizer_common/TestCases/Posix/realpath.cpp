// RUN: %clangxx -O0 %s -o %t && %run %t m1 2>&1 | FileCheck %s

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

char buff[1 << 12];
int main(int argc, char *argv[]) {
  printf("REALPATH %s\n", realpath(argv[0], buff));
  // CHECK: REALPATH /{{.+}}/realpath.cpp

  char *buff2 = realpath(argv[0], nullptr);
  printf("REALPATH %s\n", buff2);
  // CHECK: REALPATH /{{.+}}/realpath.cpp
  free(buff2);

  buff2 = realpath(".", nullptr);
  printf("REALPATH %s\n", buff2);
  // CHECK: REALPATH /{{.+}}
  free(buff2);
}
