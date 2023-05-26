// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -o - %s | FileCheck %s
// rdar://10139365

@interface Test58
- (char* &) interior __attribute__((objc_returns_inner_pointer));
- (int&)reference_to_interior_int __attribute__((objc_returns_inner_pointer));
@end

void foo() {
   Test58 *ptr;
   char *c = [(ptr) interior];

   int i = [(ptr) reference_to_interior_int];
}

// CHECK: [[T0:%.*]] = load {{.*}} {{%.*}}, align 8
// call ptr @llvm.objc.retainAutorelease(ptr [[T0]]) nounwind
// CHECK: [[T2:%.*]] = load {{.*}} {{%.*}}, align 8
// call ptr @llvm.objc.retainAutorelease(ptr [[T2]]) nounwind

