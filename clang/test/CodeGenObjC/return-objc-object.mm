// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64 -emit-llvm -o - %s | FileCheck %s

@protocol P1 @end
@interface NSOperationQueue 
{
  char ch[64];
  double d;
}
@end

NSOperationQueue &f();
NSOperationQueue<P1> &f1();

void call_once() {
  f();
  f1();
}
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0* @_Z1fv()
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) %0* @_Z2f1v()  
