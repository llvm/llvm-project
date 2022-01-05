// RUN: %clang -target mipsel-unknown-linux -S -o - -emit-llvm %s \
// RUN: | FileCheck %s

// This checks that the frontend will accept inline asm memory constraints.

int foo()
{

 // 'R': An address that can be used in a non-macro load or stor'
 // This test will result in the higher and lower nibbles being
 // switched due to the lwl/lwr instruction pairs.
 // CHECK:   %{{[0-9]+}} = call i32 asm sideeffect  "lwl $0, 1 + $1\0A\09lwr $0, 2 + $1\0A\09", "=r,*R,~{$1}"(i32* elementtype(i32) %{{[0-9,a-f]+}}) #1,

  int c = 0xffbbccdd;

  int *p = &c;
  int out = 0;

  __asm volatile (
    "lwl %0, 1 + %1\n\t"
    "lwr %0, 2 + %1\n\t"
    : "=r"(out)
    : "R"(*p)
    );
  return 0;
}
