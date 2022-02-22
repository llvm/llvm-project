// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -O1 -disable-llvm-passes | FileCheck %s --check-prefix=ALL --check-prefix=O1
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -O0 | FileCheck %s --check-prefix=ALL --check-prefix=O0

// In all tests, make sure that no expect is generated if optimizations are off.
// If optimizations are on, generate the correct expect and preserve other necessary operations.

int expect_taken(int x) {
// ALL-LABEL: define{{.*}} i32 @expect_taken
// O1:        call i64 @llvm.expect.i64(i64 {{%.*}}, i64 1)
// O0-NOT:    @llvm.expect

  if (__builtin_expect (x, 1))
    return 0;
  return x;
}


int expect_not_taken(int x) {
// ALL-LABEL: define{{.*}} i32 @expect_not_taken
// O1:        call i64 @llvm.expect.i64(i64 {{%.*}}, i64 0)
// O0-NOT:    @llvm.expect

  if (__builtin_expect (x, 0))
    return 0;
  return x;
}


int x;
int y(void);
void foo(void);

void expect_value_side_effects(void) {
// ALL-LABEL: define{{.*}} void @expect_value_side_effects()
// ALL:       [[CALL:%.*]] = call i32 @y
// O1:        [[SEXT:%.*]] = sext i32 [[CALL]] to i64
// O1:        call i64 @llvm.expect.i64(i64 {{%.*}}, i64 [[SEXT]])
// O0-NOT:    @llvm.expect

  if (__builtin_expect (x, y()))
    foo ();
}


// Make sure that issigprocmask() is called before bar()?
// There's no compare, so there's nothing to expect?
// rdar://9330105
void isigprocmask(void);
long bar(void);

int main(void) {
// ALL-LABEL: define{{.*}} i32 @main()
// ALL:       call void @isigprocmask()
// ALL:       [[CALL:%.*]] = call i64 @bar()
// O1:        call i64 @llvm.expect.i64(i64 0, i64 [[CALL]])
// O0-NOT:    @llvm.expect

  (void) __builtin_expect((isigprocmask(), 0), bar());
}


int switch_cond(int x) {
// ALL-LABEL: define{{.*}} i32 @switch_cond
// O1:        call i64 @llvm.expect.i64(i64 {{%.*}}, i64 5)
// O0-NOT:    @llvm.expect

  switch(__builtin_expect(x, 5)) {
  default:
    return 0;
  case 0:
  case 1:
  case 2:
    return 1;
  case 5:
    return 5;
  };

  return 0;
}

int variable_expected(int stuff) {
// ALL-LABEL: define{{.*}} i32 @variable_expected(
// O1: call i64 @llvm.expect.i64(i64 {{%.*}}, i64 {{%.*}})
// O0-NOT: @llvm.expect

  int res = 0;

  switch (__builtin_expect(stuff, stuff)) {
  case 0:
    res = 1;
    break;
  default:
    break;
  }

  return res;
}
