// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64 %s -S -o /dev/null -DWARN -verify
// RUN: %clang_cc1 -triple x86_64 %s -S -o /dev/null -Werror -verify
// RUN: %clang_cc1 -triple x86_64-linux-gnu %s -S -o - | FileCheck %s
void f() {
  asm("movaps %xmm3, (%esi, 2)");
// expected-note@1 {{instantiated into assembly here}}
#ifdef WARN
// expected-warning@-3 {{scale factor without index register is ignored}}
#else
// expected-error@-5 {{scale factor without index register is ignored}}
#endif
}

static unsigned var[1] = {};
void g(void) { asm volatile("movd %%xmm0, %0"
                            :
                            : "m"(var)); }

void pr40890(void) {
  struct s {
    int a, b;
  } s;
  __asm__ __volatile__("\n#define S_A abcd%0\n" : : "n"(&((struct s*)0)->a));
  __asm__ __volatile__("\n#define S_B abcd%0\n" : : "n"(&((struct s*)0)->b));
  __asm__ __volatile__("\n#define BEEF abcd%0\n" : : "n"((int*)0xdeadbeeeeeef));

// CHECK-LABEL: pr40890
// CHECK: #define S_A abcd$0
// CHECK: #define S_B abcd$4
// CHECK: #define BEEF abcd$244837814038255
}
