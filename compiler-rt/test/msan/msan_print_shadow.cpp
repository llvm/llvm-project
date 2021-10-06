// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-NO-ORIGINS < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ORIGINS < %t.out

// RUN: %clangxx_msan -fsanitize-memory-track-origins=2 -O0 -g %s -o %t && %run %t >%t.out 2>&1
// RUN: FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-ORIGINS --check-prefix=CHECK-ORIGINS-2 < %t.out

#include <sanitizer/msan_interface.h>

int main(void) {
  char volatile x;
  char *p = new char[320];
  p[2] = p[5] = 1;
  p[8] = p[9] = p[10] = p[11] = p[12] = 2;

  __msan_allocated_memory(p + 4*3, 4);
  __msan_allocated_memory(p + 4*4, 4);
  __msan_allocated_memory(p + 4*5, 4);
  __msan_allocated_memory(p + 4*6, 4);
  __msan_allocated_memory(p + 4*7, 4);
  __msan_allocated_memory(p + 4*8, 4);
  __msan_allocated_memory(p + 4*9, 4);
  __msan_allocated_memory(p + 4*10, 4);
  __msan_allocated_memory(p + 4*11, 4);
  __msan_allocated_memory(p + 4*12, 4);
  __msan_allocated_memory(p + 4*13, 4);
  __msan_allocated_memory(p + 4*14, 4);
  __msan_allocated_memory(p + 4*15, 4);
  __msan_allocated_memory(p + 4*16, 4);
  __msan_allocated_memory(p + 4*17, 4);
  __msan_allocated_memory(p + 4*18, 4);
  __msan_allocated_memory(p + 4*19, 4);
  __msan_allocated_memory(p + 4*20, 4);
  __msan_allocated_memory(p + 4*21, 4);
  __msan_allocated_memory(p + 4*22, 4);
  __msan_allocated_memory(p + 4*23, 4);
  __msan_allocated_memory(p + 4*24, 4);
  __msan_allocated_memory(p + 4*25, 4);
  __msan_allocated_memory(p + 4*26, 4);
  __msan_allocated_memory(p + 4*27, 4);
  __msan_allocated_memory(p + 4*28, 4);
  __msan_allocated_memory(p + 4*29, 4);
  __msan_allocated_memory(p + 4*30, 4);
  __msan_allocated_memory(p + 4*31, 4);

  p[19] = x;

  __msan_print_shadow(p+5, 297);
  delete[] p;
  return 0;
}

// CHECK: Shadow map [0x{{.*}}, 0x{{.*}}) of [0x{{.*}}, 0x{{.*}}), 297 bytes:

// CHECK-NO-ORIGINS: 0x{{.*}}: ..00ffff 00000000 ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff
// CHECK-NO-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffff.... ........

// CHECK-ORIGINS: 0x{{.*}}: ..00ffff 00000000 ffffffff ffffffff  |A . B C|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |D E F G|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |H I J K|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |L M N O|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |P Q R S|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |T U V W|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |X Y Z *|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |* * * A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffffffff ffffffff  |A A A A|
// CHECK-ORIGINS: 0x{{.*}}: ffffffff ffffffff ffff.... ........  |A A A .|

// CHECK-ORIGINS: Origin A (origin_id {{.*}}):
// CHECK-ORIGINS:   Uninitialized value was created by a heap allocation
// CHECK-ORIGINS:     #1 {{.*}} in main{{.*}}msan_print_shadow.cpp:14

// CHECK-ORIGINS: Origin B (origin_id {{.*}}):
// CHECK-ORIGINS:   Memory was marked as uninitialized
// CHECK-ORIGINS:     #0 {{.*}} in __msan_allocated_memory
// CHECK-ORIGINS:     #1 {{.*}} in main{{.*}}msan_print_shadow.cpp:18

// CHECK-ORIGINS: Origin C (origin_id {{.*}}):
// CHECK-ORIGINS-2:  Uninitialized value was stored to memory at
// CHECK-ORIGINS-2:    #0 {{.*}} in main{{.*}}msan_print_shadow.cpp:48
// CHECK-ORIGINS:   Uninitialized value was created by an allocation of 'x' in the stack frame of function 'main'
// CHECK-ORIGINS:     #0 {{.*}} in main{{.*}}msan_print_shadow.cpp:12

// CHECK-ORIGINS: Origin D (origin_id {{.*}}):
// CHECK-ORIGINS:   Memory was marked as uninitialized
// CHECK-ORIGINS:     #0 {{.*}} in __msan_allocated_memory
// CHECK-ORIGINS:     #1 {{.*}} in main{{.*}}msan_print_shadow.cpp:20

// ...

// CHECK-ORIGINS: Origin Z (origin_id {{.*}}):
// CHECK-ORIGINS:   Memory was marked as uninitialized
// CHECK-ORIGINS:     #0 {{.*}} in __msan_allocated_memory
// CHECK-ORIGINS:     #1 {{.*}} in main{{.*}}msan_print_shadow.cpp:42
