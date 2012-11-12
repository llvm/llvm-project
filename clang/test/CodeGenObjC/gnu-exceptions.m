// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fexceptions -fobjc-exceptions -fobjc-runtime=gcc -o - %s | FileCheck %s

void opaque(void);
void log(int i);

@class C;

// CHECK: define void @test0() {
void test0() {
  @try {
    // CHECK: invoke void @opaque()
    opaque();

    // CHECK: call void @log(i32 1)

  } @catch (C *c) {
    // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gnu_objc_personality_v0 to i8*)
    // CHECK-NEXT:   catch i8* getelementptr inbounds ([2 x i8]* @0, i64 0, i64 0)
    // CHECK:      br i1

    // CHECK: call void @log(i32 0)

    // CHECK: resume

    log(0);
  }

  log(1);
}
