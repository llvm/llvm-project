// RUN: %clangxx_tsan %s -o %t -framework Foundation -std=c++11
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#import <libkern/OSAtomic.h>

int main(int argc, const char *argv[]) {
  int value = 1;
  bool ret = OSAtomicTestAndClear(7, &value);
  fprintf(stderr, "value = %d, ret = %d\n", value, ret);
  // CHECK: value = 0, ret = 1

  ret = OSAtomicTestAndSet(4, &value);
  fprintf(stderr, "value = %d, ret = %d\n", value, ret);
  // CHECK: value = 8, ret = 0

  ret = OSAtomicTestAndClear(4, &value);
  fprintf(stderr, "value = %d, ret = %d\n", value, ret);
  // CHECK: value = 0, ret = 1

  ret = OSAtomicTestAndSet(12, &value);
  fprintf(stderr, "value = %d, ret = %d\n", value, ret);
  // CHECK: value = 2048, ret = 0

  ret = OSAtomicTestAndSet(13, &value);
  fprintf(stderr, "value = %d, ret = %d\n", value, ret);
  // CHECK: value = 3072, ret = 0

  ret = OSAtomicTestAndClear(12, &value);
  fprintf(stderr, "value = %d, ret = %d\n", value, ret);
  // CHECK: value = 1024, ret = 1

  return 0;
}
