// RUN: %clang_cc1 -triple x86_64-apple-macosx10.13.0 -std=c++1z -fobjc-arc -fobjc-exceptions -fcxx-exceptions -fexceptions -emit-llvm -o - %s | FileCheck %s

// CHECK: %[[STRUCT_CLASS1:.*]] = type { ptr }

@interface Class0;
@end

struct Class1 {
  Class0 *f;
};

struct Container {
  Class1 a;
  bool b;
};

bool getBool() {
  extern void mayThrow();
  mayThrow();
  return false;
}

Class0 *g;

// CHECK: define {{.*}} @_Z4testv()
// CHECK: invoke noundef zeroext i1 @_Z7getBoolv()
// CHECK: landingpad { ptr, i32 }
// CHECK: call void @_ZN6Class1D1Ev(ptr {{[^,]*}} %{{.*}})
// CHECK: br label

// CHECK: define linkonce_odr void @_ZN6Class1D1Ev(

Container test() {
  return {{g}, getBool()};
}
