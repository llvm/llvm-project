// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-gc -emit-llvm -o - %s | FileCheck %s

@interface I
{
  __weak id wObject;
}
@property (readwrite, weak) id representedObject;
@property (readwrite, weak) id wObject;
@property (readwrite, weak) __weak id wRandom;
@property (readwrite, assign) __weak id wAnother;
@end

@implementation I
@synthesize representedObject;
@synthesize wObject;
@synthesize wRandom;
@synthesize wAnother;
@end
// CHECK:  call ptr @objc_read_weak
// CHECK:  call ptr @objc_assign_weak
// CHECK:  call ptr @objc_read_weak
// CHECK:  call ptr @objc_assign_weak
// CHECK:  call ptr @objc_read_weak
// CHECK:  call ptr @objc_assign_weak
// CHECK:  call ptr @objc_read_weak
// CHECK:  call ptr @objc_assign_weak

