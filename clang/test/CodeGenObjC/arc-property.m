// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -emit-llvm %s -o - | FileCheck %s

// rdar://problem/10290317
@interface Test0
- (void) setValue: (id) x;
@end
void test0(Test0 *t0, id value) {
  t0.value = value;
}
// CHECK: define void @test0(
// CHECK: call i8* @objc_retain(
// CHECK: call i8* @objc_retain(
// CHECK: @objc_msgSend
// CHECK: call void @objc_storeStrong(
// CHECK: call void @objc_storeStrong(

struct S1 { Class isa; };
@interface Test1
@property (nonatomic, strong) __attribute__((NSObject)) struct S1 *pointer;
@end
@implementation Test1
@synthesize pointer;
@end
//   The getter should be a simple load.
// CHECK:    define internal [[S1:%.*]]* @"\01-[Test1 pointer]"(
// CHECK:      [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test1.pointer"
// CHECK-NEXT: [[T0:%.*]] = bitcast [[TEST1:%.*]]* {{%.*}} to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8* [[T0]], i64 [[OFFSET]]
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to [[S1]]**
// CHECK-NEXT: [[T3:%.*]] = load [[S1]]** [[T2]], align 8
// CHECK-NEXT: ret [[S1]]* [[T3]]

//   The setter should be using objc_setProperty.
// CHECK:    define internal void @"\01-[Test1 setPointer:]"(
// CHECK:      [[T0:%.*]] = bitcast [[TEST1]]* {{%.*}} to i8*
// CHECK-NEXT: [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test1.pointer"
// CHECK-NEXT: [[T1:%.*]] = load [[S1]]** {{%.*}}
// CHECK-NEXT: [[T2:%.*]] = bitcast [[S1]]* [[T1]] to i8*
// CHECK-NEXT: call void @objc_setProperty(i8* [[T0]], i8* {{%.*}}, i64 [[OFFSET]], i8* [[T2]], i1 zeroext false, i1 zeroext false)
// CHECK-NEXT: ret void


// rdar://problem/12039404
@interface Test2 {
@private
  Class _theClass;
}
@property (copy) Class theClass;
@end

static Class theGlobalClass;
@implementation Test2
@synthesize theClass = _theClass;
- (void) test {
  _theClass = theGlobalClass;
}
@end
// CHECK:    define internal void @"\01-[Test2 test]"(
// CHECK:      [[T0:%.*]] = load i8** @theGlobalClass, align 8
// CHECK-NEXT: [[T1:%.*]] = load [[TEST2:%.*]]**
// CHECK-NEXT: [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T2:%.*]] = bitcast [[TEST2]]* [[T1]] to i8*
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8* [[T2]], i64 [[OFFSET]]
// CHECK-NEXT: [[T4:%.*]] = bitcast i8* [[T3]] to i8**
// CHECK-NEXT: call void @objc_storeStrong(i8** [[T4]], i8* [[T0]]) nounwind
// CHECK-NEXT: ret void

// CHECK:    define internal i8* @"\01-[Test2 theClass]"(
// CHECK:      [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T0:%.*]] = call i8* @objc_getProperty(i8* {{.*}}, i8* {{.*}}, i64 [[OFFSET]], i1 zeroext true)
// CHECK-NEXT: ret i8* [[T0]]

// CHECK:    define internal void @"\01-[Test2 setTheClass:]"(
// CHECK:      [[T0:%.*]] = bitcast [[TEST2]]* {{%.*}} to i8*
// CHECK-NEXT: [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T1:%.*]] = load i8** {{%.*}}
// CHECK-NEXT: call void @objc_setProperty(i8* [[T0]], i8* {{%.*}}, i64 [[OFFSET]], i8* [[T1]], i1 zeroext true, i1 zeroext true)
// CHECK-NEXT: ret void

// CHECK:    define internal void @"\01-[Test2 .cxx_destruct]"(
// CHECK:      [[T0:%.*]] = load [[TEST2]]**
// CHECK-NEXT: [[OFFSET:%.*]] = load i64* @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T1:%.*]] = bitcast [[TEST2]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i8* [[T1]], i64 [[OFFSET]]
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to i8**
// CHECK-NEXT: call void @objc_storeStrong(i8** [[T3]], i8* null) nounwind
// CHECK-NEXT: ret void
