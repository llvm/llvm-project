// RUN: %clangxx %s -o %t && %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: android, hwasan, ubsan

#include <stdio.h>
#include <stdlib.h>
#if defined(__GLIBC_PREREQ) && __GLIBC_PREREQ(2, 2)
#include <mcheck.h>
#else
#define MCHECK_OK 0
extern "C" int mcheck(void (*abortfunc)(int mstatus));
extern "C" int mcheck_pedantic(void (*abortfunc)(int mstatus));
extern "C" int mprobe(void *ptr);
#endif

void check_heap() {
  void *p = malloc(1000);
  int res = mprobe(p);
  if (res == MCHECK_OK)
    printf("Success!\n");
  free(p);
}

int main(int argc, char *argv[]) {
  void *p;
  if (mcheck(NULL) != 0) {
    fprintf(stderr, "mcheck() failed\n");
    exit(EXIT_FAILURE);
  }

  check_heap();
  // CHECK: Success!

  if (mcheck_pedantic(NULL) != 0) {
    fprintf(stderr, "mcheck_pedantic() failed\n");
    exit(EXIT_FAILURE);
  }

  check_heap();
  // CHECK: Success!

  return 0;
}
