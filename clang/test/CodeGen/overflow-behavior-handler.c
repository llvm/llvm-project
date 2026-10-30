// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,NOSAN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -foverflow-behavior-label=efault -emit-llvm -o - %s | FileCheck %s --check-prefix=RENAMED
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fsanitize=signed-integer-overflow -fsanitize-recover=signed-integer-overflow -emit-llvm -o - %s | FileCheck %s --check-prefix=RECOVER
// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -fsanitize=signed-integer-overflow -fsanitize-trap=signed-integer-overflow -emit-llvm -o - %s | FileCheck %s --check-prefix=SANTRAP

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
