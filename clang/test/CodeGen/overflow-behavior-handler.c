// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -foverflow-behavior-label=efault -emit-llvm -o - %s | FileCheck %s --check-prefix=RENAMED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fsanitize=signed-integer-overflow -fsanitize-recover=signed-integer-overflow -emit-llvm -o - %s | FileCheck %s --check-prefix=RECOVER
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -emit-llvm -o - %s | FileCheck %s --check-prefix=SANTRAP
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-recover=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change -emit-llvm -o - %s | FileCheck %s --check-prefix=CONVRECOVER
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fsanitize=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change -fsanitize-trap=implicit-unsigned-integer-truncation,implicit-signed-integer-truncation,implicit-integer-sign-change -emit-llvm -o - %s | FileCheck %s --check-prefix=CONVTRAP

// CHECK-LABEL: define {{.*}} @add(
// CHECK: [[R:%.*]] = call { i32, i1 } @llvm.sadd.with.overflow.i32
// CHECK: [[OF:%.*]] = extractvalue { i32, i1 } [[R]], 1
// CHECK: br i1 [[OF]], label %[[H:[a-z.0-9]+]], label %[[CONT:[a-z.0-9]+]]
// CHECK: [[H]]:
// CHECK-NEXT: br label %__overflow
// CHECK: [[CONT]]:
// CHECK-NEXT: store i32
// CHECK: __overflow:
// CHECK-NOT: ubsantrap

// RENAMED-LABEL: define {{.*}} @add(
// RENAMED: call void @llvm.ubsantrap

// RECOVER-LABEL: define {{.*}} @add(
// RECOVER: call void @__ubsan_handle_add_overflow(
// RECOVER: br label %__overflow

// SANTRAP-LABEL: define {{.*}} @add(
// SANTRAP-NOT: ubsantrap
// SANTRAP: br label %__overflow
// SANTRAP-LABEL: define {{.*}} @inc(
int add(int __ob_trap a) {
  a = a + 1;
  return a;
__overflow:
  return -1;
}

// CHECK-LABEL: define {{.*}} @inc(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %__overflow
int inc(int __ob_trap a) {
  a++;
  return a;
__overflow:
  return -1;
}

// CHECK-LABEL: define {{.*}} @mul_assign(
// CHECK: @llvm.umul.with.overflow.i32
// CHECK: br label %__overflow
unsigned mul_assign(unsigned __ob_trap a) {
  a *= 3;
  return a;
__overflow:
  return 0;
}

// CHECK-LABEL: define {{.*}} @neg(
// CHECK: @llvm.ssub.with.overflow.i32
// CHECK: br label %__overflow
int neg(int __ob_trap a) {
  return -a;
__overflow:
  return -1;
}

// No handler label: trap as before.
// CHECK-LABEL: define {{.*}} @no_handler(
// NOSAN: call void @llvm.ubsantrap
int no_handler(int __ob_trap a) {
  return a + 1;
}

// RENAMED-LABEL: define {{.*}} @renamed(
// RENAMED: br label %efault
int renamed(int __ob_trap a) {
  return a + 1;
efault:
  return -1;
}

// A GNU local label takes over inside its block.
// CHECK-LABEL: define {{.*}} @local_label(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %[[LOCAL:__overflow[0-9]*]]
// CHECK: [[LOCAL]]:
// CHECK-NEXT: store i32 -2
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %[[OUTER:__overflow[0-9]*]]
// CHECK: [[OUTER]]:
// CHECK-NEXT: store i32 -1
int local_label(int __ob_trap a) {
  int r = ({
    __label__ __overflow;
    int t = a * 2;
    goto done;
  __overflow:
    t = -2;
  done:
    t;
  });
  return r + a;
__overflow:
  return -1;
}

void fr(int *);

// Leaving a cleanup scope runs the cleanup before reaching the handler.
// CHECK-LABEL: define {{.*}} @out_of_cleanup(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: call void @fr(
// CHECK: __overflow:
int out_of_cleanup(int __ob_trap n) {
  {
    int z __attribute__((cleanup(fr))) = 0;
    n *= 2;
  }
  return n;
__overflow:
  return -1;
}

// Implicit truncation into a plain type branches before the store.
// CHECK-LABEL: define {{.*}} @trunc_assign(
// CHECK: [[T:%.*]] = trunc i64 {{.*}} to i8
// CHECK: [[OK:%.*]] = icmp eq i64
// CHECK: [[BAD:%.*]] = xor i1 [[OK]], true
// CHECK: br i1 [[BAD]], label %[[H:[a-z.0-9]+]], label %[[CONT:[a-z.0-9]+]]
// CHECK: [[H]]:
// CHECK-NEXT: br label %__overflow
// CHECK: [[CONT]]:
// CHECK-NEXT: store i8 [[T]]
// CHECK-NOT: ubsantrap
// CHECK: ret

// CONVRECOVER-LABEL: define {{.*}} @trunc_assign(
// CONVRECOVER: call void @__ubsan_handle_implicit_conversion(
// CONVRECOVER: br label %__overflow

// CONVTRAP-LABEL: define {{.*}} @trunc_assign(
// CONVTRAP: trunc i64 {{.*}} to i8
// CONVTRAP-NOT: ubsantrap
// CONVTRAP: br label %__overflow
// CONVTRAP-LABEL: define {{.*}} @sign_change(
int trunc_assign(unsigned long long __ob_trap v) {
  unsigned char old = 5;
  old = v;
  return old;
__overflow:
  return -old;
}

// CHECK-LABEL: define {{.*}} @sign_change(
// CHECK: icmp slt i32
// CHECK: br label %__overflow
// CHECK-NOT: ubsantrap
// CHECK: ret
// CONVRECOVER-LABEL: define {{.*}} @sign_change(
// CONVRECOVER: call void @__ubsan_handle_implicit_conversion(
// CONVRECOVER: br label %__overflow
unsigned sign_change(int __ob_trap v) {
  unsigned u = v;
  return u;
__overflow:
  return 0;
}

// The conversion back to the left-hand type of a compound assignment is
// checked even when the operator itself is not.
// CHECK-LABEL: define {{.*}} @shift_assign(
// CHECK: shl i32
// CHECK: trunc i32 {{.*}} to i8
// CHECK: br label %__overflow
// CHECK-NOT: ubsantrap
// CHECK: ret
int shift_assign(unsigned char __ob_trap x) {
  x <<= 4;
  return x;
__overflow:
  return -1;
}

