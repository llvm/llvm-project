// RUN: %clang_cc1 -triple arm64-apple-macosx -emit-llvm -O0 -o - %s | FileCheck %s

// An FP-affecting pragma in effect over an Objective-C method body must put that
// body into strict-FP mode, exactly as it does for a plain function body.

#pragma STDC FENV_ACCESS ON

__attribute__((objc_root_class))
@interface Foo
@end

@implementation Foo
// CHECK-LABEL: define internal float @"\01-[Foo add:with:]"
// CHECK-SAME:  #[[ATTR:[0-9]+]]
// CHECK:       call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
- (float)add:(float)a with:(float)b {
  return a + b;
}
@end

// CHECK-LABEL: define{{.*}} float @plain_function
// CHECK-SAME:  #[[ATTR]]
// CHECK:       call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
float plain_function(float a, float b) {
  return a + b;
}

// CHECK: attributes #[[ATTR]] = {{.*}} strictfp
