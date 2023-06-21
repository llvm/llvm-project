// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fexceptions -o - %s | FileCheck %s
// rdar://8409336

struct TFENode {
void GetURL() const;
};

@interface TNodeIconAndNameCell
- (const TFENode&) node;
@end

@implementation TNodeIconAndNameCell     
- (const TFENode&) node {
// CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @objc_msgSend
// CHECK-NEXT: call void @_ZNK7TFENode6GetURLEv(ptr {{[^,]*}} %{{.*}})
	self.node.GetURL();
}	// expected-warning {{non-void function does not return a value}}
@end

// rdar://8437240
struct X {
  int x;
};

void f0(const X &parent);
@interface A
- (const X&) target;
@end
void f1(A *a) {
// CHECK: [[PRP:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @objc_msgSend
// CHECK-NEXT:call void @_Z2f0RK1X(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[PRP]])
  f0(a.target);

// CHECK: [[MSG:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @objc_msgSend
// CHECK-NEXT:call void @_Z2f0RK1X(ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[MSG]])
  f0([a target]);
}

@interface Test2
@property (readonly) int myProperty;
- (int) myProperty;
- (double) myGetter;
@end
void test2() {
    Test2 *obj;
    (void) obj.myProperty; 
    (void) obj.myGetter; 
    static_cast<void>(obj.myProperty);
    static_cast<void>(obj.myGetter);
    void(obj.myProperty);
    void(obj.myGetter);
}
// CHECK-LABEL: define{{.*}} void @_Z5test2v()
// CHECK: call noundef i32
// CHECK: call noundef double
// CHECK: call noundef i32
// CHECK: call noundef double
// CHECK: call noundef i32
// CHECK: call noundef double

// PR8751
int test3(Test2 *obj) { return obj.myProperty; }
