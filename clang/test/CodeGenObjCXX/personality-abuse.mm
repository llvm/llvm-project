// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

extern "C" {
  int __objc_personality_v0();
}

void *abuse_personality_func() {
  return (void *)&__objc_personality_v0;
}

void foo() {
  try {
    throw 0;
  } catch (int e) {
    return;
  }
}

// CHECK: define void @_Z3foov() {{#[0-9]+}} personality i8* bitcast (i32 ()* @__objc_personality_v0 to i8*)
