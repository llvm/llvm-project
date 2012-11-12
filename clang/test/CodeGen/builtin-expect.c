// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

int x;
int y(void);
void foo();
void FUNC() {
// CHECK: [[call:%.*]] = call i32 @y
  if (__builtin_expect (x, y()))
    foo ();
}

// rdar://9330105
void isigprocmask(void);
long bar();

int main() {
    (void) __builtin_expect((isigprocmask(), 0), bar());
}

// CHECK: call void @isigprocmask()
// CHECK: [[C:%.*]] = call i64 (...)* @bar()


// CHECK: @test1
int test1(int x) {
// CHECK: @llvm.expect
  if (__builtin_expect (x, 1))
    return 0;
  return x;
}

// CHECK: @test2
int test2(int x) {
// CHECK: @llvm.expect
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
