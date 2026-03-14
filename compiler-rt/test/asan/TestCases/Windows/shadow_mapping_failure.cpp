// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s
// REQUIRES: asan-32-bits

#include <stdio.h>

char bigchunk[1 << 30];

int main() {
  printf("Hello, world!\n");
  scanf("%s", bigchunk);
// CHECK-NOT: Hello, world!
// CHECK: Shadow memory range interleaves with an existing memory mapping.
// CHECK: ASan shadow was supposed to be located in the [0x2fff0000-0x{{.*}}ffff] range.
// CHECK: Dumping process modules:
// CHECK-DAG: 0x{{[0-9a-f]*}}-0x{{[0-9a-f]*}} {{.*}}shadow_mapping_failure
// CHECK-DAG: 0x{{[0-9a-f]*}}-0x{{[0-9a-f]*}} {{.*}}ntdll.dll
}
