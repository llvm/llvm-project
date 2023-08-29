// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-arc -emit-llvm %s -o - | FileCheck %s

@interface Test0
- (void) setValue: (id) x;
@end
void test0(Test0 *t0, id value) {
  t0.value = value;
}
// CHECK-LABEL: define{{.*}} void @test0(
// CHECK: call void @llvm.objc.storeStrong
// CHECK: call void @llvm.objc.storeStrong
// CHECK: @objc_msgSend
// CHECK: call void @llvm.objc.storeStrong(
// CHECK: call void @llvm.objc.storeStrong(

struct S1 { Class isa; };
@interface Test1
@property (nonatomic, strong) __attribute__((NSObject)) struct S1 *pointer;
@end
@implementation Test1
@synthesize pointer;
@end
//   The getter should be a simple load.
// CHECK:    define internal ptr @"\01-[Test1 pointer]"(
// CHECK:      [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test1.pointer"
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, ptr {{%.*}}, i64 [[OFFSET]]
// CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[T1]], align 8
// CHECK-NEXT: ret ptr [[T3]]

//   The setter should be using objc_setProperty.
// CHECK:    define internal void @"\01-[Test1 setPointer:]"(
// CHECK: [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test1.pointer"
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: call void @objc_setProperty(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef [[OFFSET]], ptr noundef [[T1]], i1 noundef zeroext false, i1 noundef zeroext false)
// CHECK-NEXT: ret void


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
// CHECK:      [[T0:%.*]] = load ptr, ptr @theGlobalClass, align 8
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds i8, ptr [[T1]], i64 [[OFFSET]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[T3]], ptr [[T0]]) [[NUW:#[0-9]+]]
// CHECK-NEXT: ret void

// CHECK:    define internal ptr @"\01-[Test2 theClass]"(
// CHECK:      [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T0:%.*]] = tail call ptr @objc_getProperty(ptr noundef {{.*}}, ptr noundef {{.*}}, i64 noundef [[OFFSET]], i1 noundef zeroext true)
// CHECK-NEXT: ret ptr [[T0]]

// CHECK:    define internal void @"\01-[Test2 setTheClass:]"(
// CHECK: [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T1:%.*]] = load ptr, ptr {{%.*}}
// CHECK-NEXT: call void @objc_setProperty(ptr noundef {{%.*}}, ptr noundef {{%.*}}, i64 noundef [[OFFSET]], ptr noundef [[T1]], i1 noundef zeroext true, i1 noundef zeroext true)
// CHECK-NEXT: ret void

// CHECK:    define internal void @"\01-[Test2 .cxx_destruct]"(
// CHECK:      [[T0:%.*]] = load ptr, ptr
// CHECK-NEXT: [[OFFSET:%.*]] = load i64, ptr @"OBJC_IVAR_$_Test2._theClass"
// CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds i8, ptr [[T0]], i64 [[OFFSET]]
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[T2]], ptr null) [[NUW]]
// CHECK-NEXT: ret void

@interface Test3
@property id copyMachine;
@end

void test3(Test3 *t) {
  id x = t.copyMachine;
  x = [t copyMachine];
}
// CHECK:    define{{.*}} void @test3(ptr
//   Prologue.
// CHECK:      [[T:%.*]] = alloca ptr,
// CHECK-NEXT: [[X:%.*]] = alloca ptr,
//   Property access.
// CHECK:      [[T0:%.*]] = load ptr, ptr [[T]],
// CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T2:%.*]] = call ptr @objc_msgSend(ptr noundef [[T0]], ptr noundef [[SEL]])
// CHECK-NEXT: store ptr [[T2]], ptr [[X]],
//   Message send.
// CHECK-NEXT: [[T0:%.*]] = load ptr, ptr [[T]],
// CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES
// CHECK-NEXT: [[T2:%.*]] = call ptr @objc_msgSend(ptr noundef [[T0]], ptr noundef [[SEL]])
// CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[X]],
// CHECK-NEXT: store ptr [[T2]], ptr [[X]],
// CHECK-NEXT: call void @llvm.objc.release(ptr [[T3]])
//   Epilogue.
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[X]], ptr null)
// CHECK-NEXT: call void @llvm.objc.storeStrong(ptr [[T]], ptr null)
// CHECK-NEXT: ret void

@implementation Test3
- (id) copyMachine {
  extern id test3_helper(void);
  return test3_helper();
}
// CHECK:    define internal ptr @"\01-[Test3 copyMachine]"(
// CHECK:      [[T0:%.*]] = call ptr @test3_helper()
// CHECK-NEXT: [[T1:%.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[T0]])
// CHECK-NEXT: ret ptr [[T1]]
- (void) setCopyMachine: (id) x {}
@end

// When synthesizing a property that's declared in multiple protocols, ensure
// that the setter is emitted if any of these declarations is readwrite.
@protocol ABC
@property (copy, nonatomic,  readonly) Test3 *someId;
@end
@protocol ABC__Mutable <ABC>
@property (copy, nonatomic, readwrite) Test3 *someId;
@end

@interface ABC_Class <ABC, ABC__Mutable>
@end

@implementation ABC_Class
@synthesize someId = _someId;
// CHECK:  define internal ptr @"\01-[ABC_Class someId]"
// CHECK:  define internal void @"\01-[ABC_Class setSomeId:]"(
@end


// CHECK: attributes [[NUW]] = { nounwind }
