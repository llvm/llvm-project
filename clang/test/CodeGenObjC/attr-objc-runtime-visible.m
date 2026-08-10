// RUN: %clang_cc1 -triple x86_64-apple-darwin -fobjc-runtime=macosx-10.9.0 -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -triple i386-apple-darwin -fobjc-runtime=macosx-fragile-10.9.0 -emit-llvm %s -o - | FileCheck %s

@interface Root
+(Class)class;
@end

__attribute__((objc_runtime_visible))
__attribute__((objc_runtime_name("MyRuntimeVisibleClass")))
@interface A : Root
@end

// CHECK: [[CLASSNAME:@.*]] = private unnamed_addr constant [22 x i8] c"MyRuntimeVisibleClass
// CHECK: define{{.*}} ptr @getClass() #0 {
Class getClass(void) {
  // CHECK: call ptr @objc_lookUpClass(ptr [[CLASSNAME]]) #2
  return [A class];
}
