// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NO-ORIGINS < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ORIGINS < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ORIGINS < %t.out

#include <sanitizer/msan_interface.h>

int main(void) {
  char *p = new char[16];
  __msan_print_shadow(p, 1);
  __msan_print_shadow(p+1, 1);
  __msan_print_shadow(p+3, 1);
  __msan_print_shadow(p+15, 1);
  __msan_print_shadow(p, 0);
  delete[] p;
  int x = 0;
  __msan_print_shadow(&x, 3);
  return 0;
}

// CHECK: Shadow map [0x{{.*}}, 0x{{.*}}) of [0x{{.*}}, 0x{{.*}}), 1 bytes:
// CHECK-NO-ORIGINS:   0x{{.*}}: ff...... ........ ........ ........
// CHECK-ORIGINS:   0x{{.*}}: ff...... ........ ........ ........  |A . . .|
// CHECK-ORIGINS: Origin A (origin_id {{.*}}):

// CHECK: Shadow map [0x{{.*}}, 0x{{.*}}) of [0x{{.*}}, 0x{{.*}}), 1 bytes:
// CHECK-NO-ORIGINS:   0x{{.*}}: ..ff.... ........ ........ ........
// CHECK-ORIGINS:   0x{{.*}}: ..ff.... ........ ........ ........  |A . . .|
// CHECK-ORIGINS: Origin A (origin_id {{.*}}):

// CHECK: Shadow map [0x{{.*}}, 0x{{.*}}) of [0x{{.*}}, 0x{{.*}}), 1 bytes:
// CHECK-NO-ORIGINS:   0x{{.*}}: ......ff ........ ........ ........
// CHECK-ORIGINS:   0x{{.*}}: ......ff ........ ........ ........  |A . . .|
// CHECK-ORIGINS: Origin A (origin_id {{.*}}):

// CHECK: Shadow map [0x{{.*}}, 0x{{.*}}) of [0x{{.*}}, 0x{{.*}}), 1 bytes:
// CHECK-NO-ORIGINS:   0x{{.*}}: ......ff ........ ........ ........
// CHECK-ORIGINS:   0x{{.*}}: ......ff ........ ........ ........  |A . . .|
// CHECK-ORIGINS: Origin A (origin_id {{.*}}):

// CHECK: Shadow map [0x{{.*}}, 0x{{.*}}) of [0x{{.*}}, 0x{{.*}}), 0 bytes:

// CHECK: Shadow map [0x{{.*}}, 0x{{.*}}) of [0x{{.*}}, 0x{{.*}}), 3 bytes:
// CHECK-NO-ORIGINS:   0x{{.*}}: 000000.. ........ ........ ........
// CHECK-ORIGINS:   0x{{.*}}: 000000.. ........ ........ ........  |. . . .|
