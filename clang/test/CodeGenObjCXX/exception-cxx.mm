// RUN: %clang_cc1 -no-opaque-pointers -x objective-c++ -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

// rdar://problem/22155434
namespace test0 {
  void foo() {
    try {
      throw 0;
    } catch (int e) {
      return;
    }
  }
// CHECK: define{{.*}} void @_ZN5test03fooEv() #0 personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
}
