// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s -check-prefix=UNINIT
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO

template<typename T> void used(T &) noexcept;

extern "C" {

extern int get_int(int) noexcept;
struct C {
  int x;
  int y;
};
extern C make_c() noexcept;

// Scalar with a self-reference: does need auto-init.
// UNINIT-LABEL:  test_selfinit_call(
// ZERO-LABEL:    test_selfinit_call(
// ZERO: store i32 0, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_selfinit_call(
// PATTERN: store i32 -1431655766, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
void test_selfinit_call() {
  int self = get_int(self);
  used(self);
}

// Scalar without a self-reference: no auto-init needed.
// UNINIT-LABEL:  test_nonself_call(
// ZERO-LABEL:    test_nonself_call(
// ZERO-NOT: !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_nonself_call(
// PATTERN-NOT: !annotation [[AUTO_INIT:!.+]]
void test_nonself_call() {
  int x = get_int(2);
  used(x);
}

// Scalar with a self-reference: does need auto-init.
// UNINIT-LABEL:  test_selfinit_lambda_call(
// ZERO-LABEL:    test_selfinit_lambda_call(
// ZERO: store i32 0, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_selfinit_lambda_call(
// PATTERN: store i32 -1431655766, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
void test_selfinit_lambda_call() {
  int self = [&](){ return self; }();
  used(self);
}

// Scalar with a self-reference: does need auto-init.
// UNINIT-LABEL:  test_selfinit_gnu_stmt_expression(
// ZERO-LABEL:    test_selfinit_gnu_stmt_expression(
// ZERO: store i32 0, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_selfinit_gnu_stmt_expression(
// PATTERN: store i32 -1431655766, ptr %self, align 4, !annotation [[AUTO_INIT:!.+]]
void test_selfinit_gnu_stmt_expression() {
  int self = ({int x = self; x + 1; });
  used(self);
}

// Not a scalar: auto-init just in case
// UNINIT-LABEL:  test_nonscalar_call(
// ZERO-LABEL:    test_nonscalar_call(
// ZERO: call void @llvm.memset{{.*}}, i8 0, i64 8, {{.*}} !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_nonscalar_call(
// PATTERN: call void @llvm.memcpy{{.*}}, i64 8, {{.*}} !annotation [[AUTO_INIT:!.+]]
void test_nonscalar_call() {
  C c = make_c();
  used(c);
}

// Scalar with a self-reference: does need auto-init.
// UNINIT-LABEL:  test_self_ptr(
// ZERO-LABEL:    test_self_ptr(
// ZERO: store ptr null, ptr %self, align 8, !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_self_ptr(
// PATTERN: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %self, align 8, !annotation [[AUTO_INIT:!.+]]
void test_self_ptr() {
  void* self = self;
  used(self);
}

// Scalar without a self-reference: no auto-init needed.
// UNINIT-LABEL:  test_nonself_ptr(
// ZERO-LABEL:    test_nonself_ptr(
// ZERO-NOT: !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_nonself_ptr(
// PATTERN-NOT: !annotation [[AUTO_INIT:!.+]]
void test_nonself_ptr() {
  int y = 0;
  void* x = &y;
  used(x);
}

// Scalar with a self-reference: does need auto-init.
// UNINIT-LABEL:  test_self_complex(
// ZERO-LABEL:    test_self_complex(
// ZERO: call void @llvm.memset{{.*}} !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_self_complex(
// PATTERN: call void @llvm.memcpy{{.*}} !annotation [[AUTO_INIT:!.+]]
void test_self_complex() {
  _Complex float self = 3.0 * 3.0 * self;
  used(self);
}

// Scalar without a self-reference: no auto-init needed.
// UNINIT-LABEL:  test_nonself_complex(
// ZERO-LABEL:    test_nonself_complex(
// ZERO-NOT: !annotation [[AUTO_INIT:!.+]]
// PATTERN-LABEL: test_nonself_complex(
// PATTERN-NOT: !annotation [[AUTO_INIT:!.+]]
void test_nonself_complex() {
  _Complex float y = 0.0;
  _Complex float x = 3.0 * 3.0 * y;
  used(x);
}

} // extern "C"

// ZERO: [[AUTO_INIT]] = !{!"auto-init"}
// PATTERN: [[AUTO_INIT]] = !{!"auto-init"}

