// RUN: %clang_cc1 -triple i386-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-exceptions -fobjc-runtime=macosx-fragile-10.10 -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple i386-apple-darwin10 -emit-llvm -fblocks -fobjc-arc -fobjc-exceptions -fobjc-runtime=macosx-fragile-10.10 -o - %s | FileCheck %s -check-prefix=GLOBALS

@class Opaque;

@interface Root {
  Class isa;
}
@end

@interface A : Root {
  Opaque *strong;
  __weak Opaque *weak;
}
@end

// GLOBALS-LABEL: @OBJC_METACLASS_A
//  Strong layout: scan the first word.
// GLOBALS: @OBJC_CLASS_NAME_{{.*}} = private unnamed_addr constant [2 x i8] c"\01\00"
//  Weak layout: skip the first word, scan the second word.
// GLOBALS: @OBJC_CLASS_NAME_{{.*}} = private unnamed_addr constant [2 x i8] c"\11\00"

//  0x04002001
//     ^ is compiled by ARC (controls interpretation of layouts)
//        ^ has C++ structors (no distinction for zero-initializable)
//           ^ factory (always set on non-metaclasses)
// GLOBALS: @OBJC_CLASS_A = private global {{.*}} i32 67117057

@implementation A
// CHECK-LABEL: define internal void @"\01-[A testStrong]"
// CHECK:      [[SELFVAR:%.*]] = alloca [[A:%.*]]*, align 4
- (void) testStrong {
// CHECK:      [[X:%.*]] = alloca [[OPAQUE:%.*]]*, align 4
// CHECK:      [[SELF:%.*]] = load [[A]]*, [[A]]** [[SELFVAR]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i32 4
// CHECK-NEXT: [[IVAR:%.*]] = bitcast i8* [[T1]] to [[OPAQUE]]**
// CHECK-NEXT: [[T0:%.*]] = load [[OPAQUE]]*, [[OPAQUE]]** [[IVAR]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[OPAQUE]]* [[T0]] to i8*
// CHECK-NEXT: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[OPAQUE]]*
// CHECK-NEXT: store [[OPAQUE]]* [[T3]], [[OPAQUE]]** [[X]]
  Opaque *x = strong;
// CHECK-NEXT: [[VALUE:%.*]] = load [[OPAQUE]]*, [[OPAQUE]]** [[X]]
// CHECK-NEXT: [[SELF:%.*]] = load [[A]]*, [[A]]** [[SELFVAR]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i32 4
// CHECK-NEXT: [[IVAR:%.*]] = bitcast i8* [[T1]] to [[OPAQUE]]**
// CHECK-NEXT: [[T0:%.*]] = bitcast [[OPAQUE]]** [[IVAR]] to i8**
// CHECK-NEXT: [[T1:%.*]] = bitcast [[OPAQUE]]* [[VALUE]] to i8*
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[T0]], i8* [[T1]])
  strong = x;
// CHECK-NEXT: [[T0:%.*]] = bitcast [[OPAQUE]]** [[X]] to i8**
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[T0]], i8* null)
// CHECK-NEXT: ret void
}

// CHECK-LABEL: define internal void @"\01-[A testWeak]"
// CHECK:      [[SELFVAR:%.*]] = alloca [[A]]*, align 4
- (void) testWeak {
// CHECK:      [[X:%.*]] = alloca [[OPAQUE]]*, align 4
// CHECK:      [[SELF:%.*]] = load [[A]]*, [[A]]** [[SELFVAR]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i32 8
// CHECK-NEXT: [[IVAR:%.*]] = bitcast i8* [[T1]] to [[OPAQUE]]**
// CHECK-NEXT: [[T0:%.*]] = bitcast [[OPAQUE]]** [[IVAR]] to i8**
// CHECK-NEXT: [[T1:%.*]] = call i8* @llvm.objc.loadWeakRetained(i8** [[T0]])
// CHECK-NEXT: [[T2:%.*]] = bitcast i8* [[T1]] to [[OPAQUE]]*
// CHECK-NEXT: store [[OPAQUE]]* [[T2]], [[OPAQUE]]** [[X]]
  Opaque *x = weak;
// CHECK-NEXT: [[VALUE:%.*]] = load [[OPAQUE]]*, [[OPAQUE]]** [[X]]
// CHECK-NEXT: [[SELF:%.*]] = load [[A]]*, [[A]]** [[SELFVAR]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i32 8
// CHECK-NEXT: [[IVAR:%.*]] = bitcast i8* [[T1]] to [[OPAQUE]]**
// CHECK-NEXT: [[T0:%.*]] = bitcast [[OPAQUE]]** [[IVAR]] to i8**
// CHECK-NEXT: [[T1:%.*]] = bitcast [[OPAQUE]]* [[VALUE]] to i8*
// CHECK-NEXT: call i8* @llvm.objc.storeWeak(i8** [[T0]], i8* [[T1]])
  weak = x;
// CHECK-NEXT: [[T0:%.*]] = bitcast [[OPAQUE]]** [[X]] to i8**
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[T0]], i8* null)
// CHECK-NEXT: ret void
}

// CHECK-LABEL: define internal void @"\01-[A .cxx_destruct]"
// CHECK:      [[SELFVAR:%.*]] = alloca [[A]]*, align 4
// CHECK:      [[SELF:%.*]] = load [[A]]*, [[A]]** [[SELFVAR]]
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i32 8
// CHECK-NEXT: [[IVAR:%.*]] = bitcast i8* [[T1]] to [[OPAQUE]]**
// CHECK-NEXT: [[T0:%.*]] = bitcast [[OPAQUE]]** [[IVAR]] to i8**
// CHECK-NEXT: call void @llvm.objc.destroyWeak(i8** [[T0]])
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[SELF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i8, i8* [[T0]], i32 4
// CHECK-NEXT: [[IVAR:%.*]] = bitcast i8* [[T1]] to [[OPAQUE]]**
// CHECK-NEXT: [[T0:%.*]] = bitcast [[OPAQUE]]** [[IVAR]] to i8**
// CHECK-NEXT: call void @llvm.objc.storeStrong(i8** [[T0]], i8* null)
// CHECK-NEXT: ret void
@end

// Test case for corner case of ivar layout.
@interface B : A {
  char _b_flag;
}
@end

@interface C : B {
  char _c_flag;
  __unsafe_unretained id c_unsafe[5];
  id c_strong[4];
  __weak id c_weak[3];
  id c_strong2[7];
}
@end
@implementation C @end

//  Note that these layouts implicitly start at the end of the previous
//  class rounded up to pointer alignment.
// GLOBALS-LABEL: @OBJC_METACLASS_C
//  Strong layout: skip five, scan four, skip three, scan seven
//    'T' == 0x54, '7' == 0x37
// GLOBALS: @OBJC_CLASS_NAME_{{.*}} = private unnamed_addr constant [3 x i8] c"T7\00"
//  Weak layout: skip nine, scan three
// GLOBALS: @OBJC_CLASS_NAME_{{.*}} = private unnamed_addr constant [2 x i8] c"\93\00"

extern void useBlock(void (^block)(void));

//  256 == 0x100 == starts with 1 strong
// GLOBALS: @"__block_descriptor{{.*}} = linkonce_odr hidden {{.*}}, i32 256 }
void testBlockLayoutStrong(id x) {
  useBlock(^{ (void) x; });
}

//  1   == 0x001 == starts with 1 weak
// GLOBALS: @"__block_descriptor{{.*}} = linkonce_odr hidden {{.*}}, i32 1 }
void testBlockLayoutWeak(__weak id x) {
  useBlock(^{ (void) x; });
}

// CHECK-LABEL: define{{.*}} void @testCatch()
// CHECK: [[X:%.*]] = alloca [[A:%.*]]*, align 4
// CHECK: [[Y:%.*]] = alloca i8*, align 4
// CHECK: call void @objc_exception_try_enter
// CHECK: br i1
// CHECK: call void @checkpoint(i32 noundef 0)
// CHECK: call void @objc_exception_try_exit
// CHECK: br label
// CHECK: call void @checkpoint(i32 noundef 3)
// CHECK: [[EXN:%.*]] = call i8* @objc_exception_extract
// CHECK: call i32 @objc_exception_match(
// CHECK: br i1
// CHECK: [[T0:%.*]] = bitcast i8* [[EXN]] to [[A]]*
// CHECK: [[T1:%.*]] = bitcast [[A]]* [[T0]] to i8*
// CHECK: [[T2:%.*]] = call i8* @llvm.objc.retain(i8* [[T1]])
// CHECK: [[T3:%.*]] = bitcast i8* [[T2]] to [[A]]*
// CHECK: store [[A]]* [[T3]], [[A]]** [[X]]
// CHECK: call void @checkpoint(i32 noundef 1)
// CHECK: [[T0:%.*]] = bitcast [[A]]** [[X]] to i8**
// CHECK: call void @llvm.objc.storeStrong(i8** [[T0]], i8* null)
// CHECK: br label
// CHECK: [[T0:%.*]] = call i8* @llvm.objc.retain(i8* [[EXN]])
// CHECK: store i8* [[T0]], i8** [[Y]]
// CHECK: call void @checkpoint(i32 noundef 2)
// CHECK: call void @llvm.objc.storeStrong(i8** [[Y]], i8* null)
extern void checkpoint(int n);
void testCatch(void) {
  @try {
    checkpoint(0);
  } @catch (A *x) {
    checkpoint(1);
  } @catch (id y) {
    checkpoint(2);
  }
  checkpoint(3);
}
