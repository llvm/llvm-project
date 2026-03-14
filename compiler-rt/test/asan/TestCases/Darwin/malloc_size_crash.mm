// RUN: %clang_asan %s -o %t -framework Foundation
// RUN: %run %t 2>&1 | FileCheck %s

#import <Foundation/Foundation.h>
#include <malloc/malloc.h>

int main(int argc, char *argv[]) {
  id obj = @0;
  fprintf(stderr, "obj = %p\n", obj);
  size_t size = malloc_size(obj);
  fprintf(stderr, "size = 0x%zx\n", size);
  fprintf(stderr, "Done.\n");
  // CHECK: Done.
  return 0;
}
