// RUN: %clang_cc1 -triple x86_64-linux-gnu -fexperimental-overflow-behavior-types -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -Wno-overflow-behavior-attribute-ignored -o - %s | FileCheck %s --check-prefix=OFF

// CHECK-LABEL: define {{.*}} @redirect(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %__overflow
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %too_big
// CHECK-NOT: ubsantrap
// CHECK: ret

// OFF-LABEL: define {{.*}} @redirect(
// OFF-NOT: with.overflow
// OFF: ret
int redirect(int __ob_trap a) {
  a = a * 2;
  a = __builtin_overflow_goto(a + 1, too_big);
  return a;
__overflow:
  return -1;
too_big:
  return -2;
}

// The builtin applies without a default handler, and only to its operand.
// CHECK-LABEL: define {{.*}} @only_operand(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %l
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: call void @llvm.ubsantrap
int only_operand(int __ob_trap a) {
  a = __builtin_overflow_goto(a + 1, l);
  return a * 3;
l:
  return 0;
}

// Innermost wins.
// CHECK-LABEL: define {{.*}} @nested(
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: br label %inner
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %outer
int nested(int __ob_trap a) {
  return __builtin_overflow_goto(__builtin_overflow_goto(a * 2, inner) + 1,
                                 outer);
inner:
  return -1;
outer:
  return -2;
}

// Conversions inside the operand are redirected too.
// CHECK-LABEL: define {{.*}} @conversion(
// CHECK: trunc i64 {{.*}} to i8
// CHECK: br label %narrow
// CHECK-NOT: ubsantrap
// CHECK: ret
unsigned char conversion(unsigned long __ob_trap v) {
  return __builtin_overflow_goto((unsigned char)v, narrow);
narrow:
  return 0;
}

// A void statement expression guards a whole block.
// CHECK-LABEL: define {{.*}} @stmt_block(
// CHECK: @llvm.ssub.with.overflow.i32
// CHECK: br label %l
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: br label %l
// CHECK-NOT: ubsantrap
// CHECK: ret
int stmt_block(int __ob_trap a, int c) {
  __builtin_overflow_goto(({
    a--;
    if (c)
      a += 2;
  }), l);
  return a;
l:
  return -1;
}

// Leaving the statement expression runs its cleanups.
// CHECK-LABEL: define {{.*}} @stmt_cleanup(
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: {{^}}overflow.handler:
// CHECK: store i32 [[DEST:[0-9]+]], ptr %cleanup.dest.slot
// CHECK-NEXT: br label %cleanup
// CHECK: {{^}}cleanup:
// CHECK-NEXT: call void @fr(
// CHECK: switch i32 {{.*}}, label %unreachable [
// CHECK: i32 [[DEST]], label %l
void fr(int *);
void stmt_cleanup(int __ob_trap a) {
  __builtin_overflow_goto(({
    int __attribute__((cleanup(fr))) x = 0;
    a += x;
  }), l);
l:;
}
