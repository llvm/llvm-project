// RUN: %clang_cl_asan -Od %p/dll_host.cc -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: %run %t %t.dll | FileCheck %s

#include <malloc.h>
#include <stdio.h>

extern "C" __declspec(dllexport)
int test_function() {
  int *p = (int*)malloc(1024 * sizeof(int));
  p[512] = 0;
  free(p);

  p = (int*)malloc(128);
  p = (int*)realloc(p, 2048 * sizeof(int));
  p[1024] = 0;
  free(p);

  p = (int*)calloc(16, sizeof(int));
  if (p[8] != 0)
    return 1;
  p[15]++;
  if (16 * sizeof(int) != _msize(p))
    return 2;
  free(p);

  p = new int;
  *p = 42;
  delete p;

  p = new int[42];
  p[15]++;
  delete [] p;

  printf("All ok\n");
// CHECK: All ok

  return 0;
}
