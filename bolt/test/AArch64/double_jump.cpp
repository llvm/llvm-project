// A contrived example to test the double jump removal peephole.

// RUN: %clang %cflags -O0 %s -o %t.exe
// RUN: llvm-bolt %t.exe -o %t.bolt --peepholes=double-jumps | \
// RUN:   FileCheck %s -check-prefix=CHECKBOLT
// RUN: llvm-objdump -d %t.bolt | FileCheck %s

// CHECKBOLT: BOLT-INFO: Peephole: 1 double jumps patched.

// CHECK: <_Z3foom>:
// CHECK-NEXT: sub     sp, sp, #16
// CHECK-NEXT: str     x0, [sp, #8]
// CHECK-NEXT: ldr     [[REG:x[0-28]+]], [sp, #8]
// CHECK-NEXT: cmp     [[REG]], #0
// CHECK-NEXT: b.eq    {{.*}} <_Z3foom+0x34>
// CHECK-NEXT: add     [[REG]], [[REG]], #1
// CHECK-NEXT: add     [[REG]], [[REG]], #1
// CHECK-NEXT: cmp     [[REG]], #2
// CHECK-NEXT: b.eq    {{.*}} <_Z3foom+0x28>
// CHECK-NEXT: add     [[REG]], [[REG]], #1
// CHECK-NEXT: mov     [[REG]], x1
// CHECK-NEXT: ldr     x1, [sp]
// CHECK-NEXT: b       {{.*}} <bar>
// CHECK-NEXT: ldr     x1, [sp]
// CHECK-NEXT: add     [[REG]], [[REG]], #1
// CHECK-NEXT: b       {{.*}} <bar>

extern "C" unsigned long bar(unsigned long count) { return count + 1; }

unsigned long foo(unsigned long count) {
  asm volatile("     cmp %0,#0\n"
               "     b.eq .L7\n"
               "     add %0, %0, #1\n"
               "     b .L1\n"
               ".L1: b .L2\n"
               ".L2: add  %0, %0, #1\n"
               "     cmp  %0, #2\n"
               "     b.ne .L3\n"
               "     b .L4\n"
               ".L3: b .L5\n"
               ".L5: add %0, %0, #1\n"
               ".L4: mov %0,x1\n"
               "     ldr x1, [sp]\n"
               "     b .L6\n"
               ".L7: ldr x1, [sp]\n"
               "     add %0, %0, #1\n"
               "     b .L6\n"
               ".L6: b bar\n"
               :
               : "r"(count)
               :);
  return count;
}

extern "C" int _start() { return foo(38); }
