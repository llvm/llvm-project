// This test consolidates tests for basic functionality, stub dispatch, and thunk generation
// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple arm64-apple-darwin10 \
// RUN:   -fobjc-direct-precondition-thunk %s -o - | FileCheck %s

struct my_complex_struct {
  int a, b;
};

struct my_aggregate_struct {
  int a, b;
  char buf[128];
};

__attribute__((objc_root_class))
@interface Root
- (int)getInt __attribute__((objc_direct));
@property(direct, readonly) int intProperty;
@property(direct, readonly) int intProperty2;
@property(direct, readonly) id objectProperty;
@end

@implementation Root
// CHECK-LABEL: define hidden i32 @"-[Root intProperty2]"(ptr noundef %self)
- (int)intProperty2 {
  return 42;
}

// CHECK-LABEL: define hidden i32 @"-[Root getInt]"(ptr noundef %self)
- (int)getInt __attribute__((objc_direct)) {
  return 42;
}

// CHECK-LABEL: define hidden i32 @"+[Root classGetInt]"(ptr noundef %self)
+ (int)classGetInt __attribute__((objc_direct)) {
  return 42;
}

// CHECK-LABEL: define hidden i64 @"-[Root getComplex]"(ptr noundef %self)
- (struct my_complex_struct)getComplex __attribute__((objc_direct)) {
  struct my_complex_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define hidden i64 @"+[Root classGetComplex]"(ptr noundef %self)
+ (struct my_complex_struct)classGetComplex __attribute__((objc_direct)) {
  struct my_complex_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define hidden void @"-[Root getAggregate]"(ptr {{.*}} sret(%struct.my_aggregate_struct) {{.*}} %agg.result, ptr noundef %self)
- (struct my_aggregate_struct)getAggregate __attribute__((objc_direct)) {
  struct my_aggregate_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define hidden void @"+[Root classGetAggregate]"(ptr {{.*}} sret(%struct.my_aggregate_struct) {{.*}} %agg.result, ptr noundef %self)
+ (struct my_aggregate_struct)classGetAggregate __attribute__((objc_direct)) {
  struct my_aggregate_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define hidden void @"-[Root accessCmd]"(ptr noundef %self)
- (void)accessCmd __attribute__((objc_direct)) {
  // loading the _cmd selector
  SEL sel = _cmd;
}

@end

// CHECK-LABEL: define hidden i32 @"-[Root intProperty]"(ptr noundef %self)
// CHECK-LABEL: define hidden ptr @"-[Root objectProperty]"(ptr noundef %self)

@interface Foo : Root {
  id __strong _cause_cxx_destruct;
}
@property(nonatomic, readonly, direct) int getDirect_setDynamic;
@property(nonatomic, readonly) int getDynamic_setDirect;
@end

@interface Foo ()
@property(nonatomic, readwrite) int getDirect_setDynamic;
@property(nonatomic, readwrite, direct) int getDynamic_setDirect;
- (int)directMethodInExtension __attribute__((objc_direct));
@end

@interface Foo (Cat)
- (int)directMethodInCategory __attribute__((objc_direct));
@end

__attribute__((objc_direct_members))
@implementation Foo
// CHECK-LABEL: define hidden i32 @"-[Foo directMethodInExtension]"(ptr noundef %self)
- (int)directMethodInExtension {
  return 42;
}

// CHECK-LABEL: define hidden i32 @"-[Foo getDirect_setDynamic]"(ptr noundef %self)
// CHECK-LABEL: define internal void @"\01-[Foo setGetDirect_setDynamic:]"(ptr noundef %self, ptr noundef %_cmd, i32 noundef %getDirect_setDynamic)
// CHECK-LABEL: define internal i32 @"\01-[Foo getDynamic_setDirect]"(ptr noundef %self, ptr noundef %_cmd)
// CHECK-LABEL: define hidden void @"-[Foo setGetDynamic_setDirect:]"(ptr noundef %self, i32 noundef %getDynamic_setDirect)

@end

@implementation Foo (Cat)
// CHECK-LABEL: define hidden i32 @"-[Foo directMethodInCategory]"(ptr noundef %self)
- (int)directMethodInCategory {
  return 42;
}

// CHECK-LABEL: define hidden i32 @"-[Foo directMethodInCategoryNoDecl]"(ptr noundef %self)
- (int)directMethodInCategoryNoDecl __attribute__((objc_direct)) {
  return 42;
}

@end

// CHECK-LABEL: define{{.*}} i32 @useRoot(ptr noundef %r)
int useRoot(Root *r) {
  // CHECK: call i32 @"-[Root getInt]_thunk"(ptr noundef %{{[0-9]+}})
  // CHECK: call i32 @"-[Root intProperty]_thunk"(ptr noundef %{{[0-9]+}})
  // CHECK: call i32 @"-[Root intProperty2]_thunk"(ptr noundef %{{[0-9]+}})
  return [r getInt] + [r intProperty] + [r intProperty2];
}

// Thunks are emitted after the first function that uses them
// CHECK-LABEL: define linkonce_odr hidden i32 @"-[Root getInt]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[Root getInt]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

// CHECK-LABEL: define linkonce_odr hidden i32 @"-[Root intProperty]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[Root intProperty]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

// CHECK-LABEL: define linkonce_odr hidden i32 @"-[Root intProperty2]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[Root intProperty2]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

// CHECK-LABEL: define{{.*}} i32 @useFoo(ptr noundef %f)
int useFoo(Foo *f) {
  // CHECK: call void @"-[Foo setGetDynamic_setDirect:]_thunk"(ptr noundef %{{[0-9]+}}, i32 noundef 1)
  // CHECK: call i32 @"-[Foo getDirect_setDynamic]_thunk"(ptr noundef %{{[0-9]+}})
  // CHECK: call i32 @"-[Foo directMethodInExtension]_thunk"(ptr noundef %{{[0-9]+}})
  // CHECK: call i32 @"-[Foo directMethodInCategory]_thunk"(ptr noundef %{{[0-9]+}})
  // CHECK: call i32 @"-[Foo directMethodInCategoryNoDecl]_thunk"(ptr noundef %{{[0-9]+}})
  [f setGetDynamic_setDirect:1];
  return [f getDirect_setDynamic] +
         [f directMethodInExtension] +
         [f directMethodInCategory] +
         [f directMethodInCategoryNoDecl];
}

// CHECK-LABEL: define linkonce_odr hidden void @"-[Foo setGetDynamic_setDirect:]_thunk"(ptr noundef %self, i32 noundef
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   musttail call void @"-[Foo setGetDynamic_setDirect:]"(ptr noundef %self, i32 noundef
// CHECK:   ret void
// CHECK: dummy_ret_block:
// CHECK:   ret void

// CHECK-LABEL: define linkonce_odr hidden i32 @"-[Foo getDirect_setDynamic]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[Foo getDirect_setDynamic]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

// CHECK-LABEL: define linkonce_odr hidden i32 @"-[Foo directMethodInExtension]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[Foo directMethodInExtension]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

// CHECK-LABEL: define linkonce_odr hidden i32 @"-[Foo directMethodInCategory]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[Foo directMethodInCategory]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

// CHECK-LABEL: define linkonce_odr hidden i32 @"-[Foo directMethodInCategoryNoDecl]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[Foo directMethodInCategoryNoDecl]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

__attribute__((objc_root_class))
@interface RootDeclOnly
@property(direct, readonly) int intProperty;
@end

// CHECK-LABEL: define{{.*}} i32 @useRootDeclOnly(ptr noundef %r)
int useRootDeclOnly(RootDeclOnly *r) {
  // CHECK: call i32 @"-[RootDeclOnly intProperty]_thunk"(ptr noundef %{{[0-9]+}})
  return [r intProperty];
}

// Verify thunk is generated for external direct method
// CHECK: declare{{.*}} i32 @"-[RootDeclOnly intProperty]"(ptr)
// CHECK-LABEL: define linkonce_odr hidden i32 @"-[RootDeclOnly intProperty]_thunk"(ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   %[[RET:.*]] = musttail call i32 @"-[RootDeclOnly intProperty]"(ptr noundef %self)
// CHECK:   ret i32 %[[RET]]
// CHECK: dummy_ret_block:

int useSRet(Root *r) {
  return (
    // First call is to instance method - uses thunk
    // CHECK: call i64 @"-[Root getComplex]_thunk"
    [r getComplex].a +
    // TODO: we should know that this instance is non nil.
    // CHECK: call void @"-[Root getAggregate]_thunk"
    [r getAggregate].a +
    // TODO: The compiler is not smart enough to know the class object must be realized yet.
    // CHECK: call i64 @"+[Root classGetComplex]_thunk"(ptr noundef
    [Root classGetComplex].a +
    // CHECK: call void @"+[Root classGetAggregate]_thunk"(ptr {{.*}}sret
    [Root classGetAggregate].a
  );
}

// CHECK-LABEL: define linkonce_odr hidden i64 @"-[Root getComplex]_thunk"
// CHECK-LABEL: define linkonce_odr hidden void @"-[Root getAggregate]_thunk"(ptr dead_on_unwind writable sret(%struct.my_aggregate_struct) {{.*}} %agg.result, ptr noundef %self)
// CHECK: entry:
// CHECK:   %[[IS_NIL:.*]] = icmp eq ptr {{.*}}, null
// CHECK:   br i1 %[[IS_NIL]], label %objc_direct_method.self_is_nil, label %objc_direct_method.cont
// CHECK: objc_direct_method.self_is_nil:
// CHECK:   call void @llvm.memset
// CHECK:   br label %dummy_ret_block
// CHECK: objc_direct_method.cont:
// CHECK:   musttail call void @"-[Root getAggregate]"(ptr dead_on_unwind writable sret(%struct.my_aggregate_struct) {{.*}} %agg.result, ptr noundef %self)
// CHECK:   ret void
// CHECK: dummy_ret_block:
// CHECK:   ret void

// CHECK: define {{.*}} @"+[Root classGetComplex]_thunk"
// CHECK: define {{.*}} @"+[Root classGetAggregate]_thunk"
