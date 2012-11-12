// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fdebugger-support %s -emit-llvm -o - | FileCheck %s

// rdar://problem/9416370
void test0(id x) {
  struct A { int w, x, y, z; };
  struct A result = (struct A) [x makeStruct];
  // CHECK:     define void @test0(
  // CHECK:      [[X:%.*]] = alloca i8*, align 8
  // CHECK-NEXT: [[RESULT:%.*]] = alloca [[A:%.*]], align 4
  // CHECK-NEXT: store i8* {{%.*}}, i8** [[X]],
  // CHECK-NEXT: [[T0:%.*]] = load i8** [[X]],
  // CHECK-NEXT: [[T1:%.*]] = load i8** @"\01L_OBJC_SELECTOR_REFERENCES_"
  // CHECK-NEXT: [[T2:%.*]] = call { i64, i64 } bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to { i64, i64 } (i8*, i8*)*)(i8* [[T0]], i8* [[T1]])
}
