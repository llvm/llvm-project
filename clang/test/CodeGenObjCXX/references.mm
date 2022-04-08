// RUN: %clang_cc1 -no-opaque-pointers %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

struct A { ~A(); };

@interface B {
  A a;
}

- (const A&)getA;
@end

@implementation B 

- (const A&)getA {
  return a;
}

@end

// CHECK-LABEL: define{{.*}} void @_Z1fP1B
// CHECK: objc_msgSend to
// CHECK-NOT: call void @_ZN1AD1Ev
// CHECK: ret void
void f(B* b) {
  (void)[b getA];
}

// PR7741
@protocol P1 @end
@protocol P2 @end
@protocol P3 @end
@interface foo<P1> {} @end
@interface bar : foo <P1, P2, P3> {} @end
typedef bar baz;
void f5(foo&);
void f5b(foo<P1>&);
void f5c(foo<P2>&);
void f5d(foo<P3>&);
void f6(baz* x) { 
  f5(*x); 
  f5b(*x); 
  f5c(*x); 
  f5d(*x);
  (void)((foo&)*x);  
}
