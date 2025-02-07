// RUN: %clang_cc1 -fobjc-export-direct-methods -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -O0 -o - | FileCheck %s
// RUN: %clang_cc1 -fobjc-export-direct-methods -emit-llvm -triple x86_64-apple-darwin10 %s -O0 -o - | FileCheck --check-prefix=NO-ARC %s

// If objc-arc is not set, we should not emit any arc related intrinsics.
// NO-ARC-NOT: retainAutoreleasedReturnValue
// NO-ARC-NOT: objc_retainAutoreleasedReturnValue
// NO-ARC-NOT: call void asm sideeffect "mov\09fp, fp\09\09// marker for objc_retainAutoreleaseReturnValue", ""()
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
@property(direct, readonly) Root* objectProperty;
@end

@implementation Root
// CHECK-LABEL: define dso_local i32 @"-<Root intProperty2>_inner"(ptr noundef nonnull %{{.*}}
// CHECK-LABEL: define dso_local i32 @"-<Root intProperty2>"(ptr noundef %{{.*}}
- (int)intProperty2 {
  return 42;
}

// CHECK-LABEL: define dso_local i32 @"-<Root getInt>_inner"(ptr noundef nonnull %{{.*}}
// CHECK-NEXT: entry:
// CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
// CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],
// CHECK-NEXT: ret i32 42
- (int)getInt __attribute__((objc_direct)) {
  // loading parameters
  // CHECK: define dso_local i32 @"-<Root getInt>"(ptr noundef [[ARG0:%.*]])
  // CHECK-NEXT: entry:
  // CHECK-NEXT: [[RETVAL:%.*]] = alloca
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],

  // self nil-check
  // CHECK-NEXT: [[SELF:%.*]] = load ptr, ptr [[SELFADDR]],
  // CHECK-NEXT: [[NILCHECK:%.*]] = icmp eq ptr [[SELF]], null
  // CHECK-NEXT: br i1 [[NILCHECK]],

  // setting return value to nil
  // CHECK-LABEL: objc_direct_method.self_is_nil:
  // CHECK-NEXT: call void @llvm.memset{{[^(]*}}({{[^,]*}}[[RETVAL]], i8 0,
  // CHECK-NEXT: br label

  // set value
  // CHECK-LABEL: objc_direct_method.cont:
  // CHECK-NEXT: [[RET:%.*]] = call i32 @"-<Root getInt>_inner"(ptr noundef nonnull [[ARG0]]
  // CHECK-NEXT: store i32 [[RET]], ptr [[RETVAL]]
  // CHECK-NEXT: br label %return

  // return
  // CHECK-LABEL: return:
  // CHECK-NEXT: {{%.*}} = load{{.*}}[[RETVAL]],
  // CHECK-NEXT: ret
  return 42;
}

// CHECK-NOT: @"+<Root classGetInt>_inner"
+ (int)classGetInt __attribute__((objc_direct)) {
  // CHECK: define dso_local i32 @"+<Root classGetInt>"(ptr noundef nonnull [[ARGSELF:%.*]])
  // [self self]
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr [[ARGSELF]], ptr [[SELFADDR]],
  // CHECK-NEXT: [[SELF:%.*]] = load ptr, ptr [[SELFADDR]],
  // CHECK-NEXT: [[SELFSEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[CALL:%.*]] = call {{.*}} @objc_msgSend(ptr noundef [[SELF]], ptr noundef [[SELFSEL]])
  // CHECK-NEXT: store ptr [[CALL]], ptr [[SELFADDR]],
  // CHECK-NEXT: ret i32 42
  return 42;
}

// CHECK-LABEL: define dso_local i64 @"-<Root getComplex>_inner"(ptr noundef nonnull %{{.*}}
// CHECK-LABEL: entry:
// CHECK-NEXT: [[RETVAL:%.*]] = alloca
// CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
// CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],
// CHECK-NEXT: call void @llvm.memcpy{{[^(]*}}({{[^,]*}}[[RETVAL]],
// CHECK-NEXT: [[RET:%.*]] = load{{.*}}[[RETVAL]],
// CHECK-NEXT: ret i64 [[RET]]
- (struct my_complex_struct)getComplex __attribute__((objc_direct)) {

  // CHECK: define dso_local i64 @"-<Root getComplex>"(ptr noundef [[ARGSELF:%.*]])
  // self nil-check
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[RETVAL:%.*]] = alloca
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr [[ARGSELF]], ptr [[SELFADDR]],
  // CHECK-NEXT: [[SELF:%.*]] = load ptr, ptr [[SELFADDR]],
  // CHECK-NEXT: [[NILCHECK:%.*]] = icmp eq ptr [[SELF]], null
  // CHECK-NEXT: br i1 [[NILCHECK]],

  // setting return value to nil
  // CHECK-LABEL: objc_direct_method.self_is_nil:
  // CHECK-NEXT: call void @llvm.memset{{[^(]*}}({{[^,]*}}[[RETVAL]], i8 0,
  // CHECK-NEXT: br label

  // call the inner function set value
  // CHECK-LABEL: objc_direct_method.cont:
  // CHECK-NEXT: [[CALL:%.*]] = call i64 @"-<Root getComplex>_inner"(ptr noundef nonnull [[ARGSELF]])
  // CHECK-NEXT: store i64 [[CALL]], ptr [[RETVAL]]
  // CHECK-NEXT: br label

  // return
  // CHECK-LABEL: return:
  // CHECK-NEXT: [[RET:%.*]] = load{{.*}}[[RETVAL]],
  // CHECK-NEXT: ret i64 [[RET]]
  struct my_complex_struct st = {.a = 42};
  return st;
}

// CHECK-NOT: @"+<Root classGetComplex>_inner"
+ (struct my_complex_struct)classGetComplex __attribute__((objc_direct)) {
  // CHECK-LABEL: define dso_local i64 @"+<Root classGetComplex>"(ptr noundef
  struct my_complex_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define dso_local void @"-<Root getAggregate>_inner"(
// CHECK: ptr {{.*}} sret(%struct.my_aggregate_struct) align 4 [[RETVAL:%[^,]*]], ptr noundef nonnull %self
// CHECK-LABEL: entry:
// CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
// CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],
// CHECK-NEXT: call void @llvm.memset{{[^(]*}}({{[^,]*}}[[RETVAL]], i8 0,
// CHECK-NEXT: [[A:%.*]] = getelementptr {{.*}} %struct.my_aggregate_struct, ptr [[RETVAL]], i32 0, i32 0
// CHECK-NEXT: store i32 42, ptr [[A]]
// CHECK-NEXT: ret void
- (struct my_aggregate_struct)getAggregate __attribute__((objc_direct)) {

  // loading parameters
  // CHECK-LABEL: define dso_local void @"-<Root getAggregate>"(
  // CHECK: ptr {{.*}} sret(%struct.my_aggregate_struct) align 4 [[RETVAL:%[^,]*]], ptr noundef [[ARGSELF:%.*]])
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],
  // self nil-check
  // CHECK-NEXT: [[SELF:%.*]] = load ptr, ptr [[SELFADDR]],
  // CHECK-NEXT: [[NILCHECK:%.*]] = icmp eq ptr [[SELF]], null
  // CHECK-NEXT: br i1 [[NILCHECK]],

  // setting return value to nil
  // CHECK-LABEL: objc_direct_method.self_is_nil:
  // CHECK-NEXT: call void @llvm.memset{{[^(]*}}({{[^,]*}}[[RETVAL]], i8 0,
  // CHECK-NEXT: br label

  // set value
  // CHECK-LABEL: objc_direct_method.cont:
  // CHECK-NEXT: call void @"-<Root getAggregate>_inner"(ptr dead_on_unwind noalias writable sret(%struct.my_aggregate_struct) align 4 [[RETVAL]], ptr noundef nonnull [[ARGSELF]])
  // CHECK-NEXT: br label

  // return
  // CHECK-LABEL: return:
  // CHECK-NEXT: ret void
  struct my_aggregate_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define dso_local void @"+<Root classGetAggregate>"({{.*}}, ptr noundef nonnull {{.*}})
// CHECK-NOT: @"+<Root classGetAggregate>_inner"
+ (struct my_aggregate_struct)classGetAggregate __attribute__((objc_direct)) {
  struct my_aggregate_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define dso_local void @"-<Root accessCmd>_inner"(ptr noundef nonnull
// CHECK-LABEL: entry:
// CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
// CHECK-NEXT: [[CMDVAL:%_cmd]] = alloca ptr,
// CHECK-NEXT: [[SELVAL:%sel]] = alloca ptr,
// CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],
// loading the _cmd selector
// CHECK-NEXT: [[CMD1:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
// CHECK-NEXT: store ptr [[CMD1]], ptr [[CMDVAL]],
// CHECK-NEXT: [[SEL:%.*]] = load ptr, ptr [[CMDVAL]],
// CHECK-NEXT: store ptr [[SEL]], ptr [[SELVAL]],
// CHECK-NEXT: ret void
- (void)accessCmd __attribute__((objc_direct)) {
  // CHECK-LABEL: define dso_local void @"-<Root accessCmd>"(ptr noundef %{{.*}})

  // CHECK-LABEL: objc_direct_method.self_is_nil:
  // There is nothing for us to initialize, so this is an empty block
  // CHECK-NEXT: br label %return

  // CHECK-LABEL: objc_direct_method.cont:
  // CHECK-NEXT: call void @"-<Root accessCmd>_inner"(ptr noundef nonnull %{{.*}})
  // CHECK-NEXT: br label %return
  SEL sel = _cmd;
}

@end
// CHECK-LABEL: define dso_local i32 @"-<Root intProperty>_inner"(ptr noundef nonnull %{{.*}})
// CHECK-LABEL: define dso_local i32 @"-<Root intProperty>"(ptr noundef %{{.*}})

// Check the synthesized objectProperty calls objc_getProperty(); this also
// checks that the synthesized method passes undef for the `cmd` argument.
// CHECK-LABEL: define dso_local ptr @"-<Root objectProperty>_inner"(ptr noundef nonnull {{%.*}})
// CHECK-NEXT: entry:
// CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
// CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],
// CHECK-NEXT: [[SELFVAL:%.*]] = load {{.*}} [[SELFADDR]],
// CHECK-NEXT: [[IVAR:%.*]] = load {{.*}} @"OBJC_IVAR_$_Root._objectProperty",
// CHECK-NEXT: [[CALL:%.*]] = tail call ptr @objc_getProperty(ptr noundef [[SELF]], ptr noundef poison, i64 noundef [[IVAR]], {{.*}})

// CHECK-LABEL: define dso_local ptr @"-<Root objectProperty>"(ptr noundef %{{.*}})
// CHECK-LABEL: entry:
// CHECK: [[RETADDR:%.*]] = alloca ptr,

// CHECK-LABEL: objc_direct_method.cont:
// CHECK-NEXT: [[RETVAL:%.*]] = call ptr @"-<Root objectProperty>_inner"
// CHECK-NEXT: [[RETAINED_RET:%.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[RETVAL]])
// CHECK-NEXT: store ptr [[RETAINED_RET]], ptr [[RETADDR]],
// CHECK-NEXT: br label %return

// CHECK-LABEL: return:
// CHECK-NEXT: [[RET:%.*]] = load ptr, ptr [[RETADDR]],
// CHECK-NEXT: [[AUTORELEASED:%.*]] = tail call ptr @llvm.objc.autoreleaseReturnValue(ptr [[RET]])
// CHECK-NEXT: ret ptr [[AUTORELEASED]]
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
// CHECK-LABEL: define dso_local i32 @"-<Foo directMethodInExtension>_inner"(
// CHECK-LABEL: define dso_local i32 @"-<Foo directMethodInExtension>"(
- (int)directMethodInExtension {
  return 42;
}
// CHECK-LABEL: define dso_local i32 @"-<Foo getDirect_setDynamic>_inner"(
// CHECK-LABEL: define dso_local i32 @"-<Foo getDirect_setDynamic>"(
// CHECK-LABEL: define internal void @"\01-[Foo setGetDirect_setDynamic:]"(
// CHECK-LABEL: define internal i32 @"\01-[Foo getDynamic_setDirect]"(
// CHECK-LABEL: define dso_local void @"-<Foo setGetDynamic_setDirect:>_inner"(
// CHECK-LABEL: define dso_local void @"-<Foo setGetDynamic_setDirect:>"(
// CHECK-LABEL: define internal void @"\01-[Foo .cxx_destruct]"(
@end

@implementation Foo (Cat)
// CHECK-LABEL: define dso_local i32 @"-<Foo directMethodInCategory>_inner"(
// CHECK-LABEL: define dso_local i32 @"-<Foo directMethodInCategory>"(
- (int)directMethodInCategory {
  return 42;
}
// CHECK-LABEL: define dso_local i32 @"-<Foo directMethodInCategoryNoDecl>_inner"(
// CHECK-LABEL: define dso_local i32 @"-<Foo directMethodInCategoryNoDecl>"(
- (int)directMethodInCategoryNoDecl __attribute__((objc_direct)) {
  return 42;
}
@end

// CHECK: define i32 @useClassMethod()
// CHECK:   call {{.*}} @"+<Root classGetInt>"
// CHECK:   call {{.*}} @"+<Root classGetComplex>"
// CHECK:   call {{.*}} @"+<Root classGetAggregate>"
int useClassMethod() {
  return [Root classGetInt] + [Root classGetComplex].a + [Root classGetAggregate].a;
}

int useRoot(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @useRoot
  // CHECK: %{{[^ ]*}} = call i32  @"-<Root getInt>"
  // CHECK: %{{[^ ]*}} = call i32  @"-<Root intProperty>"
  // CHECK: %{{[^ ]*}} = call i32  @"-<Root intProperty2>"
  return [r getInt] + [r intProperty] + [r intProperty2];
}

// Currently, we don't have analysis on nonnull attributes yet.
__attribute__((nonnull))int useFoo(const Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @useFoo
  // CHECK: call void @"-<Foo setGetDynamic_setDirect:>"
  // CHECK: %{{[^ ]*}} = call i32 @"-<Foo getDirect_setDynamic>"
  // CHECK: %{{[^ ]*}} = call i32 @"-<Foo directMethodInExtension>"
  // CHECK: %{{[^ ]*}} = call i32 @"-<Foo directMethodInCategory>"
  // CHECK: %{{[^ ]*}} = call i32 @"-<Foo directMethodInCategoryNoDecl>"
  [f setGetDynamic_setDirect:1];
  return [f getDirect_setDynamic] +
         [f directMethodInExtension] +
         [f directMethodInCategory] +
         [f directMethodInCategoryNoDecl];
}

__attribute__((objc_root_class))
@interface RootDeclOnly
@property(direct, readonly) int intProperty;
@end

// We can't be sure `r` is nonnull, so we need to call the thunk without underscore.
int useRootDeclOnly(RootDeclOnly *r) {
  // CHECK-LABEL: define{{.*}} i32 @useRootDeclOnly
  // CHECK: %{{[^ ]*}} = call i32 @"-<RootDeclOnly intProperty>"
  return [r intProperty];
}

// CHECK-LABEL: define i32 @getObjectIntProperty
int getObjectIntProperty(Root *r) {
  // CHECK: [[OBJ:%.*]] = call ptr @"-<Root objectProperty>"
  // CHECK-NEXT: [[RETAINED:%.*]] = notail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr [[OBJ]])
  // CHECK-NEXT: {{.*}} = call i32 @"-<Root intProperty>"(ptr noundef [[RETAINED]])
  // CHECK-NEXT: call void @llvm.objc.release(ptr [[RETAINED]])
    return r.objectProperty.intProperty;
}
// CHECK-LABEL: }
