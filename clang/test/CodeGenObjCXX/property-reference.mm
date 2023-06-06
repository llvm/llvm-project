// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - | FileCheck %s
// rdar://9208606

struct MyStruct {
  int x;
  int y;
  int z;
};

@interface MyClass {
  MyStruct _foo;
}

@property (assign, readwrite) const MyStruct& foo;

- (const MyStruct&) foo;
- (void) setFoo:(const MyStruct&)inFoo;
@end

void test0() {
  MyClass* myClass;
  MyStruct myStruct;

  myClass.foo = myStruct;

  const MyStruct& currentMyStruct = myClass.foo;   
}

// CHECK: [[C:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @objc_msgSend
// CHECK:   store ptr [[C]], ptr [[D:%.*]]

namespace test1 {
  struct A { A(); A(const A&); A&operator=(const A&); ~A(); };
}
@interface Test1 {
  test1::A ivar;
}
@property (nonatomic) const test1::A &prop1;
@end
@implementation Test1
@synthesize prop1 = ivar;
@end
// CHECK:    define internal noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @"\01-[Test1 prop1]"(
// CHECK:      [[SELF:%.*]] = alloca ptr, align 8
// CHECK:      [[T0:%.*]] = load ptr, ptr [[SELF]]
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 0
// CHECK-NEXT: ret ptr [[T2]]

// CHECK:    define internal void @"\01-[Test1 setProp1:]"(
// CHECK:      call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN5test11AaSERKS0_(
// CHECK-NEXT: ret void

// rdar://problem/10497174
@interface Test2
@property int prop;
@end

// The fact that these are all non-dependent is critical.
template <class T> void test2(Test2 *a) {
  int x = a.prop;
  a.prop = x;
  a.prop += x;
}
template void test2<int>(Test2*);
// CHECK-LABEL: define weak_odr void @_Z5test2IiEvP5Test2(
// CHECK: [[X:%.*]] = alloca i32,
// CHECK:      @objc_msgSend
// CHECK:      store i32 {{%.*}}, ptr [[X]],
// CHECK:      load i32, ptr [[X]],
// CHECK:      @objc_msgSend
// CHECK:      @objc_msgSend
// CHECK:      load i32, ptr [[X]],
// CHECK-NEXT: add nsw
// CHECK:      @objc_msgSend
// CHECK-NEXT: ret void

// Same as the previous test, but instantiation-dependent.
template <class T> void test3(Test2 *a) {
  int x = (sizeof(T), a).prop;
  a.prop = (sizeof(T), x);
  a.prop += (sizeof(T), x);
}
template void test3<int>(Test2*);
// CHECK-LABEL: define weak_odr void @_Z5test3IiEvP5Test2(
// CHECK: [[X:%.*]] = alloca i32,
// CHECK:      @objc_msgSend
// CHECK:      store i32 {{%.*}}, ptr [[X]],
// CHECK:      load i32, ptr [[X]],
// CHECK:      @objc_msgSend
// CHECK:      @objc_msgSend
// CHECK:      load i32, ptr [[X]],
// CHECK-NEXT: add nsw
// CHECK:      @objc_msgSend
// CHECK-NEXT: ret void
