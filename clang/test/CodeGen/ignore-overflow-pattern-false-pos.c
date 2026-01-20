// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-undefined-ignore-overflow-pattern=all %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsanitize=signed-integer-overflow,unsigned-integer-overflow -fsanitize-undefined-ignore-overflow-pattern=all -fwrapv %s -emit-llvm -o - | FileCheck %s

// Check for potential false positives from patterns that _almost_ match classic overflow-dependent or overflow-prone code patterns
extern unsigned a, b, c;

extern unsigned some(void);

// Make sure all these still have handler paths, we shouldn't be excluding
// instrumentation of any "near" patterns.
// CHECK-LABEL: close_but_not_quite
void close_but_not_quite(void) {
  // CHECK: br i1{{.*}}handler.
  if (a + b > a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a - b < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + b < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + b + 1 < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  // CHECK: br i1{{.*}}handler.
  if (a + b < a + 1)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (b >= a + b)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + a < a)
    c = 9;

  // CHECK: br i1{{.*}}handler.
  if (a + b == a)
    c = 9;

  // CHECK: br i1{{.*}}handler
  while (--a)
    some();
}

// CHECK-LABEL: function_calls
void function_calls(void) {
  // CHECK: br i1{{.*}}handler
  if (some() + b < some())
    c = 9;
}

// CHECK-LABEL: not_quite_a_negated_unsigned_const
void not_quite_a_negated_unsigned_const(void) {
  // CHECK: br i1{{.*}}handler
  a = -b;
}
