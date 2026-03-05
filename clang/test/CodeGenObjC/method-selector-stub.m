// RUN: %clang_cc1 -emit-llvm -fobjc-msgsend-selector-stubs -triple arm64-apple-ios15 %s -o - | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fobjc-msgsend-selector-stubs -triple arm64_32-apple-watchos8 %s -o - | FileCheck %s

__attribute__((objc_root_class))
@interface Root
- (int)test0;
+ (int)class0;
- (int)test1: (int)a0;
- (int)test2: (int)a0 withA: (int)a1;

@property(readonly) int intProperty;
@end

@interface Foo : Root
@end

@interface Foo ()
- (int)testSuper0;
- (int)methodInExtension;
@end

@interface Foo (Cat)
- (int)methodInCategory;
@end


// CHECK: [[TEST0_METHNAME:@OBJC_METH_VAR_NAME_[^ ]*]] = private unnamed_addr constant [6 x i8] c"test0\00", section "__TEXT,__objc_methname,cstring_literals"
// CHECK: [[TEST0_SELREF:@OBJC_SELECTOR_REFERENCES_[^ ]*]] = internal externally_initialized global ptr [[TEST0_METHNAME]], section "__DATA,__objc_selrefs,literal_pointers,no_dead_strip"

@implementation Foo

- (int)testSuper0 {
  // Super calls don't have stubs.
  // CHECK-LABEL: define{{.*}} i32 @"\01-[Foo testSuper0]"(
  // CHECK: [[SEL:%[^ ]]] = load ptr, ptr [[TEST0_SELREF]]
  // CHECK: %{{[^ ]*}}  = call i32 @objc_msgSendSuper2(ptr {{[^,]+}}, ptr {{[^,)]*}}[[SEL]])

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

int test_root_test0(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_test0(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$test0"(ptr {{[^,]+}}, ptr {{[^,)]*}}undef)
  return [r test0];
}

// CHECK: declare ptr @"objc_msgSend$test0"(ptr, ptr, ...)

int test_root_class0() {
  // CHECK-LABEL: define{{.*}} i32 @test_root_class0(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$class0"(ptr {{[^,]+}}, ptr {{[^,)]*}}undef)
  return [Root class0];
}

// CHECK: declare ptr @"objc_msgSend$class0"(ptr, ptr, ...)

int test_root_test1(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_test1(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$test1:"(ptr {{[^,]+}}, ptr {{[^,)]*}}undef, i32 {{[^,)]*}}42)
  return [r test1: 42];
}

// CHECK: declare ptr @"objc_msgSend$test1:"(ptr, ptr, ...)

int test_root_test2(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @test_root_test2(
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$test2:withA:"(ptr {{[^,]+}}, ptr {{[^,)]*}}undef, i32 {{[^,)]*}}42, i32 {{[^,)]*}}84)
  return [r test2: 42 withA: 84];

}

// CHECK: declare ptr @"objc_msgSend$test2:withA:"(ptr, ptr, ...)

int test_extension(Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @test_extension
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$methodInExtension"(ptr {{[^,]+}}, ptr {{[^,)]*}}undef)
  return [f methodInExtension];
}

// CHECK: declare ptr @"objc_msgSend$methodInExtension"(ptr, ptr, ...)

int test_category(Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @test_category
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$methodInCategory"(ptr {{[^,]+}}, ptr {{[^,)]*}}undef)
  return [f methodInCategory];
}

// CHECK: declare ptr @"objc_msgSend$methodInCategory"(ptr, ptr, ...)

int test_category_nodecl(Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @test_category_nodecl
  // CHECK: %{{[^ ]*}} = call i32 @"objc_msgSend$methodInCategoryNoDecl"(ptr {{[^,]+}}, ptr {{[^,)]*}}undef)
  return [f methodInCategoryNoDecl];
}

// CHECK: declare ptr @"objc_msgSend$methodInCategoryNoDecl"(ptr, ptr, ...)


// === Test the special case where there's no method, but only a selector.

@interface NSArray
@end;

extern void use(id);

void test_fastenum_rawsel(NSArray *array) {
  // CHECK-LABEL: define{{.*}} void @test_fastenum_rawsel
  // CHECK: %{{[^ ]*}} = call {{i32|i64}} @"objc_msgSend$countByEnumeratingWithState:objects:count:"(ptr {{[^,]+}}, ptr
  // CHECK-NOT: @objc_msgSend to
  for (id x in array)
    use(x);
}

// CHECK: declare ptr @"objc_msgSend$countByEnumeratingWithState:objects:count:"(ptr, ptr, ...)
