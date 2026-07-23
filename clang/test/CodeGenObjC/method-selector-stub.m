// RUN: %clang_cc1 -emit-llvm -fobjc-msgsend-selector-stubs -triple arm64-apple-ios15 %s -o - | FileCheck -check-prefix=CHECK -check-prefix=INST_STUB %s
// RUN: %clang_cc1 -emit-llvm -fobjc-msgsend-selector-stubs -triple arm64_32-apple-watchos8 %s -o - | FileCheck -check-prefix=CHECK -check-prefix=INST_STUB %s
// RUN: %clang_cc1 -emit-llvm -fobjc-msgsend-selector-stubs -fobjc-msgsend-class-selector-stubs -triple arm64-apple-ios15 %s -o - | FileCheck -check-prefix=CHECK -check-prefix=CLASS_STUB %s
// RUN: %clang_cc1 -emit-llvm -fobjc-msgsend-selector-stubs -fobjc-msgsend-class-selector-stubs -triple arm64_32-apple-watchos8 %s -o - | FileCheck -check-prefix=CHECK -check-prefix=CLASS_STUB %s

__attribute__((objc_root_class))
@interface Root
- (int)test0;
- (int)test$0;
+ (int)class0;
- (int)class0;
+ (int)class0$;
- (int)test1: (int)a0;
- (int)test2: (int)a0 withA: (int)a1;

@property(readonly) int intProperty;
@end

__attribute__((objc_root_class))
@interface Root2
+ (int)class0;
@end

@interface Foo : Root
@end

@interface Foo ()
- (int)testSuper0;
- (int)methodInExtension;
+ (int)classMethodInExtension;
@end

@interface Foo (Cat)
- (int)methodInCategory;
+ (int)classMethodInCategory;
@end


// CHECK: [[TEST0_METHNAME:@OBJC_METH_VAR_NAME_[^ ]*]] = private unnamed_addr constant [6 x i8] c"test0\00", section "__TEXT,__objc_methname,cstring_literals"
// CHECK: [[TEST0_SELREF:@OBJC_SELECTOR_REFERENCES_[^ ]*]] = internal externally_initialized global ptr [[TEST0_METHNAME]], section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip"

// INST_STUB-NOT: @llvm.used =
// CLASS_STUB: @llvm.used = appending global [1 x ptr] [ptr @"OBJC_CLASS_$_Foo"],

@implementation Foo

- (int)testSuper0 {
  // Super calls don't have stubs.
  // CHECK-LABEL: define{{.*}} i32 @"\01-[Foo testSuper0]"(
  // CHECK: [[SEL:%[^ ]]] = load ptr, ptr [[TEST0_SELREF]]
  // CHECK: %{{[^ ]*}} = call i32 @objc_msgSendSuper2(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,]+}}[[SEL]])

  return [super test0];
}

// CHECK-LABEL: define internal i32 @"\01-[Foo methodInExtension]"(
- (int)methodInExtension {
  return 42;
}
@end

@implementation Foo (Cat)
// CHECK-LABEL: define internal i32 @"\01-[Foo(Cat) methodInCategory]"(
- (int)methodInCategory {
  return 42;
}
// CHECK-LABEL: define internal i32 @"\01-[Foo(Cat) methodInCategoryNoDecl]"(
- (int)methodInCategoryNoDecl {
  return 42;
}
@end

__attribute__((objc_root_class,objc_runtime_name("_Foo1234")))
@interface Foo2
+(void)m0;
@end

__attribute__((objc_class_stub, objc_subclassing_restricted))
@interface Foo3
+(void)m0;
@end

__attribute__((objc_runtime_visible))
@interface Foo4
+(void)m0;
@end

int test_root_test0(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_test0(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$test0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  return [r test0];
}

// CHECK: declare ptr @"objc_msgSend$test0"(ptr, ptr, ...)

int test_root_test0_dollar(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_test0_dollar(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$test$0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  return [r test$0];
}

// CHECK: declare ptr @"objc_msgSend$test$0"(ptr, ptr, ...)

int test_root_class0() {
  // CHECK-LABEL: define{{.*}} i32 @test_root_class0(
  // INST_STUB: %{{[^ ]*}} = call i32 @"objc_msgSend$class0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  // CLASS_STUB: %{{[^ ]*}} = call i32 @"objc_msgSendClass$class0$_OBJC_CLASS_$_Root"(ptr {{[^,)]*}}poison, ptr {{[^,)]*}}undef)
  return [Root class0];
}

// INST_STUB: declare ptr @"objc_msgSend$class0"(ptr, ptr, ...)
// CLASS_STUB: declare ptr @"objc_msgSendClass$class0$_OBJC_CLASS_$_Root"(ptr, ptr, ...)

int test_root2_class0() {
  // CHECK-LABEL: define{{.*}} i32 @test_root2_class0(
  // INST_STUB: %{{[^ ]*}} = call i32 @"objc_msgSend$class0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  // CLASS_STUB: %{{[^ ]*}} = call i32 @"objc_msgSendClass$class0$_OBJC_CLASS_$_Root2"(ptr {{[^,)]*}}poison, ptr {{[^,)]*}}undef)
  return [Root2 class0];
}

// CLASS_STUB: declare ptr @"objc_msgSendClass$class0$_OBJC_CLASS_$_Root2"(ptr, ptr, ...)

int test_root_class0_inst(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_class0_inst(
  // CHECK: %[[R_ADDR:.*]] = alloca ptr,
  // CHECK: store ptr %{{.*}}, ptr %[[R_ADDR]],
  // CHECK: %[[V0:.*]] = load ptr, ptr %[[R_ADDR]],
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$class0"(ptr {{[^,]*}}%[[V0]], ptr {{[^,)]*}}undef)
  return [r class0];
}

// CLASS_STUB: declare ptr @"objc_msgSend$class0"(ptr, ptr, ...)

int test_root_class0_dollar() {
  // CHECK-LABEL: define{{.*}} i32 @test_root_class0_dollar(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$class0$"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  return [Root class0$];
}

// CHECK: declare ptr @"objc_msgSend$class0$"(ptr, ptr, ...)

int test_id_class0(id r) {
  // CHECK-LABEL: define{{.*}} i32 @test_id_class0(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$class0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  return [r class0];
}

int test_root_test1(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_test1(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$test1:"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef, i32 {{[^,)]*}}42)
  return [r test1: 42];
}

// CHECK: declare ptr @"objc_msgSend$test1:"(ptr, ptr, ...)

int test_root_test2(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_test2(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$test2:withA:"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef, i32 {{[^,)]*}}42, i32 {{[^,)]*}}84)
  return [r test2: 42 withA: 84];

}

// CHECK: declare ptr @"objc_msgSend$test2:withA:"(ptr, ptr, ...)

int test_extension(Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @test_extension
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$methodInExtension"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  return [f methodInExtension];
}

// CHECK: declare ptr @"objc_msgSend$methodInExtension"(ptr, ptr, ...)

int test_class_method_extension(void) {
  // CHECK-LABEL: define{{.*}} i32 @test_class_method_extension
  // INST_STUB: %{{[^ ]*}} = call i32 @"objc_msgSend$classMethodInExtension"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  // CLASS_STUB: %{{[^ ]*}} = call i32 @"objc_msgSendClass$classMethodInExtension$_OBJC_CLASS_$_Foo"(ptr {{[^,)]*}}poison, ptr {{[^,)]*}}undef)
  return [Foo classMethodInExtension];
}

// INST_STUB: declare ptr @"objc_msgSend$classMethodInExtension"(ptr, ptr, ...)
// CLASS_STUB: declare ptr @"objc_msgSendClass$classMethodInExtension$_OBJC_CLASS_$_Foo"(ptr, ptr, ...)

int test_category(Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @test_category
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$methodInCategory"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  return [f methodInCategory];
}

// CHECK: declare ptr @"objc_msgSend$methodInCategory"(ptr, ptr, ...)

int test_class_method_category(void) {
  // CHECK-LABEL: define{{.*}} i32 @test_class_method_category
  // INST_STUB: %{{[^ ]*}} = call i32 @"objc_msgSend$classMethodInCategory"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  // CLASS_STUB: %{{[^ ]*}} = call i32 @"objc_msgSendClass$classMethodInCategory$_OBJC_CLASS_$_Foo"(ptr {{[^,]*}}poison, ptr {{[^,)]*}}undef)
  return [Foo classMethodInCategory];
}

// INST_STUB: declare ptr @"objc_msgSend$classMethodInCategory"(ptr, ptr, ...)
// CLASS_STUB: declare ptr @"objc_msgSendClass$classMethodInCategory$_OBJC_CLASS_$_Foo"(ptr, ptr, ...)

void test_class_method_objc_runtime_name(void) {
  // CHECK-LABEL: define{{.*}} void @test_class_method_objc_runtime_name(
  // INST_STUB: call void @"objc_msgSend$m0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  // CLASS_STUB: call void @"objc_msgSendClass$m0$_OBJC_CLASS_$__Foo1234"(ptr {{[^,]*}}poison, ptr {{[^,)]*}}undef)
  [Foo2 m0];
}

void test_class_method_class_stub(void) {
  // CHECK-LABEL: define{{.*}} void @test_class_method_class_stub(
  // CHECK: call void @"objc_msgSend$m0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  [Foo3 m0];
}

void test_class_method_objc_runtime_visible(void) {
  // CHECK-LABEL: define{{.*}} void @test_class_method_objc_runtime_visible(
  // CHECK: call void @"objc_msgSend$m0"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  [Foo4 m0];
}

int test_category_nodecl(Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @test_category_nodecl
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$methodInCategoryNoDecl"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef)
  return [f methodInCategoryNoDecl];
}

// CHECK: declare ptr @"objc_msgSend$methodInCategoryNoDecl"(ptr, ptr, ...)


// === Test the special case where there's no method, but only a selector.

@interface NSArray
@end;

extern void use(id);

void test_fastenum_rawsel(NSArray *array) {
  // CHECK-LABEL: define{{.*}} void @test_fastenum_rawsel
  // CHECK: %{{[^ ]*}} = call {{i32|i64}} @"objc_msgSend$countByEnumeratingWithState:objects:count:"(ptr {{[^,]*}}%{{[^,]+}}, ptr {{[^,)]*}}undef,
  // CHECK-NOT: @objc_msgSend to
  for (id x in array)
    use(x);
}

// CHECK: declare ptr @"objc_msgSend$countByEnumeratingWithState:objects:count:"(ptr, ptr, ...)
