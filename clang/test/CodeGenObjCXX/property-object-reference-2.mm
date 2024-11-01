// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.7 -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple x86_64-unknown-freebsd -fobjc-runtime=gnustep-1.7 -emit-llvm -o - | FileCheck -check-prefix=CHECK-GNUSTEP %s

extern int DEFAULT();

struct TCPPObject
{
 TCPPObject();
 ~TCPPObject();
 TCPPObject(const TCPPObject& inObj, int i = DEFAULT());
 TCPPObject& operator=(const TCPPObject& inObj);
 int filler[64];
};


@interface MyDocument 
{
@private
 TCPPObject _cppObject;
 TCPPObject _cppObject1;
}
@property (assign, readwrite, atomic) const TCPPObject MyProperty;
@property (assign, readwrite, atomic) const TCPPObject MyProperty1;
@end

@implementation MyDocument
  @synthesize MyProperty = _cppObject;
  @synthesize MyProperty1 = _cppObject1;
@end

// CHECK-LABEL: define internal void @__copy_helper_atomic_property_(ptr noundef %0, ptr noundef %1) #
// CHECK: [[TWO:%.*]] = load ptr, ptr [[ADDR:%.*]], align 8
// CHECK: [[THREE:%.*]] = load ptr, ptr [[ADDR1:%.*]], align 8
// CHECK: [[CALL:%.*]] = call noundef i32 @_Z7DEFAULTv()
// CHECK:  call void @_ZN10TCPPObjectC1ERKS_i(ptr {{[^,]*}} [[TWO]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[THREE]], i32 noundef [[CALL]])
// CHECK:  ret void

// CHECK: define internal void @"\01-[MyDocument MyProperty]"(
// CHECK: call void @objc_copyCppObjectAtomic(ptr noundef [[AGGRESULT:%.*]], ptr noundef [[ADDPTR:%.*]], ptr noundef @__copy_helper_atomic_property_)
// CHECK: ret void

// CHECK-LABEL: define internal void @__assign_helper_atomic_property_(ptr noundef %0, ptr noundef %1) #
// CHECK: [[THREE:%.*]] = load ptr, ptr [[ADDR1:%.*]], align 8
// CHECK: [[TWO:%.*]] = load ptr, ptr [[ADDR:%.*]], align 8
// CHECK: [[CALL:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN10TCPPObjectaSERKS_(ptr {{[^,]*}} [[TWO]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[THREE]])
// CHECK:  ret void

// CHECK: define internal void @"\01-[MyDocument setMyProperty:]"(
// CHECK: call void @objc_copyCppObjectAtomic(ptr noundef [[ADDRPTR:%.*]], ptr noundef [[MYPROPERTY:%.*]], ptr noundef @__assign_helper_atomic_property_)
// CHECK: ret void

// CHECK-GNUSTEP: objc_getCppObjectAtomic
// CHECK-GNUSTEP: objc_setCppObjectAtomic
