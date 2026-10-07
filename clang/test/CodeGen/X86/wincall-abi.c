// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-msvc -o - -emit-llvm %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64apx-unknown-windows-msvc -o - -S %s | FileCheck -check-prefix=ASM %s

// wincall ABI for x86_64apx-windows targets:
//   - empty objects take no register slots
//   - aggregates <= 32 bytes are passed and returned in registers
//   - larger aggregates use sret / indirect passing

struct empty {};

__attribute__((wincall)) void take_empty(struct empty e, int x);

__attribute__((wincall)) struct empty make_empty(void);

void call_empty(void) {
  struct empty e;
  take_empty(e, 42);
  // CHECK-LABEL: define dso_local x86_wincallcc void @"\01call_empty@win"
  // CHECK: call x86_wincallcc void @"\01take_empty@win"(i32 noundef 42)
  make_empty();
  // CHECK: call x86_wincallcc void @"\01make_empty@win"()
  // CHECK: declare dso_local x86_wincallcc void @"\01take_empty@win"(i32 noundef)
  // CHECK: declare dso_local x86_wincallcc void @"\01make_empty@win"()
}

__attribute__((wincall)) void take_empty2(struct empty e, int x) {}
// CHECK-LABEL: define dso_local x86_wincallcc void @"\01take_empty2@win"(i32 noundef %x)

struct span2 {
  unsigned long long *base;
  unsigned long long len;
};

__attribute__((wincall)) struct span2 make_span(void) {
  struct span2 s = {0, 1};
  return s;
}
// CHECK-LABEL: define dso_local x86_wincallcc %struct.span2 @"\01make_span@win"
// ASM-LABEL: make_span@win:
// ASM: movq (%rsp), %rax
// ASM: movq 8(%rsp), %rdx

struct vec4 {
  unsigned long long a, b, c, d;
};

__attribute__((wincall)) struct vec4 make_vec4(void) {
  struct vec4 v = {1, 2, 3, 4};
  return v;
}
// CHECK-LABEL: define dso_local x86_wincallcc %struct.vec4 @"\01make_vec4@win"
// ASM-LABEL: make_vec4@win:
// ASM: movq (%rsp), %rax
// ASM: movq 8(%rsp), %rdx
// ASM: movq 16(%rsp), %rcx
// ASM: movq 24(%rsp), %r8

struct big32 {
  unsigned long long a[5];
};

__attribute__((wincall)) struct big32 make_big(void) {
  struct big32 b = {{1, 2, 3, 4, 5}};
  return b;
}
// CHECK-LABEL: define dso_local x86_wincallcc void @"\01make_big@win"(ptr dead_on_unwind noalias writable sret(%struct.big32) align 8 %agg.result)
// ASM-LABEL: make_big@win:

// FP and integer registers are allocated independently: a double between two
// ints does not consume or skip an integer register.
__attribute__((wincall)) void take_mixed(int a, double b, int c, double d,
                                        int e, double f) {
  volatile int sink1 = a + c + e;
  volatile double sink2 = b + d + f;
  (void)sink1;
  (void)sink2;
}
// CHECK-LABEL: define dso_local x86_wincallcc void @"\01take_mixed@win"(i32 noundef %a, double noundef %b, i32 noundef %c, double noundef %d, i32 noundef %e, double noundef %f)
void call_mixed(void) {
  take_mixed(1, 2.0, 3, 4.0, 5, 6.0);
  // CHECK-LABEL: define dso_local x86_wincallcc void @"\01call_mixed@win"
  // CHECK: call x86_wincallcc void @"\01take_mixed@win"(i32 noundef 1, double noundef 2.000000e+00, i32 noundef 3, double noundef 4.000000e+00, i32 noundef 5, double noundef 6.000000e+00)
}
// ASM-LABEL: take_mixed@win:
// ASM: movsd %xmm2, 48(%rsp)
// ASM: movl %r8d, 44(%rsp)
// ASM: movsd %xmm1, 32(%rsp)
// ASM: movl %edx, 28(%rsp)
// ASM: movsd %xmm0, 16(%rsp)
// ASM: movl %ecx, 12(%rsp)

__attribute__((wincall)) __int128 make_i128(void) { return (__int128)1 << 64 | 2; }
// ASM-LABEL: make_i128@win:
// ASM: movl $2, %eax
// ASM: movl $1, %edx
