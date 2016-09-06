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
// CHECK: define i8* @getClass() #0 {
Class getClass(void) {
  // CHECK: call i8* @objc_lookUpClass(i8* getelementptr inbounds ([22 x i8], [22 x i8]* [[CLASSNAME]], i32 0, i32 0)) #2
  return [A class];
}
