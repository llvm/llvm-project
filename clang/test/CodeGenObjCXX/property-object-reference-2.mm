// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.7 -emit-llvm -o - | FileCheck %s
// rdar://6137845

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

// CHECK: define internal void @__copy_helper_atomic_property_(
// CHECK: [[TWO:%.*]] = load %struct.TCPPObject** [[ADDR:%.*]], align 8
// CHECK: [[THREE:%.*]] = load %struct.TCPPObject** [[ADDR1:%.*]], align 8
// CHECK: [[CALL:%.*]] = call i32 @_Z7DEFAULTv()
// CHECK:  call void @_ZN10TCPPObjectC1ERKS_i(%struct.TCPPObject* [[TWO]], %struct.TCPPObject* [[THREE]], i32 [[CALL]])
// CHECK:  ret void

// CHECK: define internal void @"\01-[MyDocument MyProperty]"(
// CHECK: [[ONE:%.*]] = bitcast i8* [[ADDPTR:%.*]] to %struct.TCPPObject*
// CHECK: [[TWO:%.*]] = bitcast %struct.TCPPObject* [[ONE]] to i8*
// CHECK: [[THREE:%.*]] = bitcast %struct.TCPPObject* [[AGGRESULT:%.*]] to i8*
// CHECK: call void @objc_copyCppObjectAtomic(i8* [[THREE]], i8* [[TWO]], i8* bitcast (void (%struct.TCPPObject*, %struct.TCPPObject*)* @__copy_helper_atomic_property_ to i8*))
// CHECK: ret void

// CHECK: define internal void @__assign_helper_atomic_property_(
// CHECK: [[TWO:%.*]] = load %struct.TCPPObject** [[ADDR:%.*]], align 8
// CHECK: [[THREE:%.*]] = load %struct.TCPPObject** [[ADDR1:%.*]], align 8
// CHECK: [[CALL:%.*]] = call %struct.TCPPObject* @_ZN10TCPPObjectaSERKS_(%struct.TCPPObject* [[TWO]], %struct.TCPPObject* [[THREE]])
// CHECK:  ret void

// CHECK: define internal void @"\01-[MyDocument setMyProperty:]"(
// CHECK: [[ONE:%.*]] = bitcast i8* [[ADDRPTR:%.*]] to %struct.TCPPObject*
// CHECK: [[TWO:%.*]] = bitcast %struct.TCPPObject* [[ONE]] to i8*
// CHECK: [[THREE:%.*]] = bitcast %struct.TCPPObject* [[MYPROPERTY:%.*]] to i8*
// CHECK: call void @objc_copyCppObjectAtomic(i8* [[TWO]], i8* [[THREE]], i8* bitcast (void (%struct.TCPPObject*, %struct.TCPPObject*)* @__assign_helper_atomic_property_ to i8*))
// CHECK: ret void
