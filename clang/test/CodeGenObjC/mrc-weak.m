// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-10.10 -emit-llvm -fblocks -fobjc-weak -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-MODERN
// RUN: %clang_cc1 -triple i386-apple-darwin10 -fobjc-runtime=macosx-fragile-10.10 -emit-llvm -fblocks -fobjc-weak -o - %s | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-FRAGILE

@interface Object
- (instancetype) retain;
- (void) run;
@end

// The ivars in HighlyAlignedSubclass should be placed in the tail-padding
// of the superclass.  Ensure that they're still covered by layouts.
@interface HighlyAligned : Object {
  __attribute__((aligned(32))) void *array[2];
}
@end
// CHECK-MODERN: @"OBJC_IVAR_$_HighlyAlignedSubclass.ivar2" = global i64 24,
// CHECK-MODERN: @"OBJC_IVAR_$_HighlyAlignedSubclass.ivar" = global i64 16,
// CHECK-MODERN: @OBJC_CLASS_NAME_{{.*}} = {{.*}} c"\02\00"
// CHECK-MODERN: @"\01l_OBJC_CLASS_RO_$_HighlyAlignedSubclass" = {{.*}} {
// CHECK-FRAGILE: @OBJC_INSTANCE_VARIABLES_HighlyAlignedSubclass = {{.*}}, i32 8 }, {{.*}}, i32 12 }]
// CHECK-FRAGILE: @OBJC_CLASS_NAME_{{.*}} = {{.*}} c"\02\00"
// CHECK-FRAGILE: @OBJC_CLASS_HighlyAlignedSubclass
@interface HighlyAlignedSubclass : HighlyAligned {
  __weak id ivar;
  __weak id ivar2;
}
@end
@implementation HighlyAlignedSubclass @end

// CHECK-MODERN: @OBJC_CLASS_NAME_{{.*}} = {{.*}} c"\01\00"
// CHECK-MODERN: @"\01l_OBJC_CLASS_RO_$_Foo" = {{.*}} { i32 772
//   772 == 0x304
//            ^ HasMRCWeakIvars
//            ^ HasCXXDestructorOnly
//              ^ HasCXXStructors

// CHECK-FRAGILE: @OBJC_CLASS_NAME_{{.*}} = {{.*}} c"\01\00"
// CHECK-FRAGILE: @OBJC_CLASS_Foo = {{.*}} i32 134225921,
//   134225921 == 0x08002001
//                   ^ HasMRCWeakIvars
//                      ^ HasCXXStructors
//                         ^ Factory
@interface Foo : Object {
  __weak id ivar;
}
@end

@implementation Foo
// CHECK-LABEL: define internal void @"\01-[Foo .cxx_destruct]"
// CHECK: call void @objc_destroyWeak
@end


void test1(__weak id x) {}
// CHECK-LABEL: define void @test1
// CHECK:      [[X:%.*]] = alloca i8*,
// CHECK-NEXT: objc_initWeak
// CHECK-NEXT: objc_destroyWeak
// CHECK-NEXT: ret void

void test2(id y) {
  __weak id z = y;
}
// CHECK-LABEL: define void @test2
// CHECK:      [[Y:%.*]] = alloca i8*,
// CHECK-NEXT: [[Z:%.*]] = alloca i8*,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
// CHECK-NEXT: call i8* @objc_initWeak(i8** [[Z]], i8* [[T0]])
// CHECK-NEXT: call void @objc_destroyWeak(i8** [[Z]])
// CHECK-NEXT: ret void

void test3(id y) {
  __weak id z;
  z = y;
}
// CHECK-LABEL: define void @test3
// CHECK:      [[Y:%.*]] = alloca i8*,
// CHECK-NEXT: [[Z:%.*]] = alloca i8*,
// CHECK-NEXT: store
// CHECK-NEXT: store i8* null, i8** [[Z]]
// CHECK-NEXT: [[T0:%.*]] = load i8*, i8** [[Y]]
// CHECK-NEXT: call i8* @objc_storeWeak(i8** [[Z]], i8* [[T0]])
// CHECK-NEXT: call void @objc_destroyWeak(i8** [[Z]])
// CHECK-NEXT: ret void

void test4(__weak id *p) {
  id y = *p;
}
// CHECK-LABEL: define void @test4
// CHECK:      [[P:%.*]] = alloca i8**,
// CHECK-NEXT: [[Y:%.*]] = alloca i8*,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load i8**, i8*** [[P]]
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_loadWeak(i8** [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT: ret void

void test5(__weak id *p) {
  id y = [*p retain];
}
// CHECK-LABEL: define void @test5
// CHECK:      [[P:%.*]] = alloca i8**,
// CHECK-NEXT: [[Y:%.*]] = alloca i8*,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load i8**, i8*** [[P]]
// CHECK-NEXT: [[T1:%.*]] = call i8* @objc_loadWeakRetained(i8** [[T0]])
// CHECK-NEXT: store i8* [[T1]], i8** [[Y]]
// CHECK-NEXT: ret void

void test6(__weak Foo **p) {
  Foo *y = [*p retain];
}
// CHECK-LABEL: define void @test6
// CHECK:      [[P:%.*]] = alloca [[FOO:%.*]]**,
// CHECK-NEXT: [[Y:%.*]] = alloca [[FOO]]*,
// CHECK-NEXT: store
// CHECK-NEXT: [[T0:%.*]] = load [[FOO]]**, [[FOO]]*** [[P]]
// CHECK-NEXT: [[T1:%.*]] = bitcast [[FOO]]** [[T0]] to i8**
// CHECK-NEXT: [[T2:%.*]] = call i8* @objc_loadWeakRetained(i8** [[T1]])
// CHECK-NEXT: [[T3:%.*]] = bitcast i8* [[T2]] to [[FOO]]*
// CHECK-NEXT: store [[FOO]]* [[T3]], [[FOO]]** [[Y]]
// CHECK-NEXT: ret void

extern id get_object(void);
extern void use_block(void (^)(void));

void test7(void) {
  __weak Foo *p = get_object();
  use_block(^{ [p run ]; });
}
// CHECK-LABEL: define void @test7
// CHECK:       [[P:%.*]] = alloca [[FOO]]*,
// CHECK:       [[T0:%.*]] = call i8* @get_object()
// CHECK-NEXT:  [[T1:%.*]] = bitcast i8* [[T0]] to [[FOO]]*
// CHECK-NEXT:  [[T2:%.*]] = bitcast [[FOO]]** [[P]] to i8**
// CHECK-NEXT:  [[T3:%.*]] = bitcast [[FOO]]* [[T1]] to i8*
// CHECK-NEXT:  call i8* @objc_initWeak(i8** [[T2]], i8* [[T3]])
// CHECK:       call void @objc_copyWeak
// CHECK:       call void @use_block
// CHECK:       call void @objc_destroyWeak

// CHECK-LABEL: define internal void @__copy_helper_block
// CHECK:       @objc_copyWeak

// CHECK-LABEL: define internal void @__destroy_helper_block
// CHECK:       @objc_destroyWeak

void test8(void) {
  __block __weak Foo *p = get_object();
  use_block(^{ [p run ]; });
}
// CHECK-LABEL: define void @test8
// CHECK:       call i8* @objc_initWeak
// CHECK-NOT:   call void @objc_copyWeak
// CHECK:       call void @use_block
// CHECK:       call void @objc_destroyWeak

// CHECK-LABEL: define internal void @__Block_byref_object_copy
// CHECK:       call void @objc_moveWeak

// CHECK-LABEL: define internal void @__Block_byref_object_dispose
// CHECK:       call void @objc_destroyWeak

// CHECK-LABEL: define void @test9_baseline()
// CHECK:       define internal void @__copy_helper
// CHECK:       define internal void @__destroy_helper
void test9_baseline(void) {
  Foo *p = get_object();
  use_block(^{ [p run]; });
}

// CHECK-LABEL: define void @test9()
// CHECK-NOT:   define internal void @__copy_helper
// CHECK-NOT:   define internal void @__destroy_helper
// CHECK:       define void @test9_fin()
void test9(void) {
  __unsafe_unretained Foo *p = get_object();
  use_block(^{ [p run]; });
}
void test9_fin() {}

// CHECK-LABEL: define void @test10()
// CHECK-NOT:   define internal void @__copy_helper
// CHECK-NOT:   define internal void @__destroy_helper
// CHECK:       define void @test10_fin()
void test10(void) {
  typedef __unsafe_unretained Foo *UnsafeFooPtr;
  UnsafeFooPtr p = get_object();
  use_block(^{ [p run]; });
}
void test10_fin() {}
