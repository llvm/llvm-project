// RUN: %clang_cc1 -emit-llvm -fobjc-arc -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

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
// CHECK-LABEL: define hidden i32 @"\01-[Root intProperty2]"
- (int)intProperty2 {
  return 42;
}

// CHECK-LABEL: define hidden i32 @"\01-[Root getInt]"(
- (int)getInt __attribute__((objc_direct)) {
  // loading parameters
  // CHECK-LABEL: entry:
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
  // CHECK: store{{.*}}[[RETVAL]],
  // CHECK-NEXT: br label

  // return
  // CHECK-LABEL: return:
  // CHECK: {{%.*}} = load{{.*}}[[RETVAL]],
  // CHECK-NEXT: ret
  return 42;
}

// CHECK-LABEL: define hidden i32 @"\01+[Root classGetInt]"(
+ (int)classGetInt __attribute__((objc_direct)) {
  // loading parameters
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
  // CHECK-NEXT: store ptr %{{.*}}, ptr [[SELFADDR]],

  // [self self]
  // CHECK-NEXT: [[SELF:%.*]] = load ptr, ptr [[SELFADDR]],
  // CHECK-NEXT: [[SELFSEL:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[SELF0:%.*]] = call {{.*}} @objc_msgSend
  // CHECK-NEXT: store ptr [[SELF0]], ptr [[SELFADDR]],

  // return
  // CHECK-NEXT: ret
  return 42;
}

// CHECK-LABEL: define hidden i64 @"\01-[Root getComplex]"(
- (struct my_complex_struct)getComplex __attribute__((objc_direct)) {
  // loading parameters
  // CHECK-LABEL: entry:
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
  // CHECK-NEXT: call void @llvm.memcpy{{[^(]*}}({{[^,]*}}[[RETVAL]],
  // CHECK-NEXT: br label

  // return
  // CHECK-LABEL: return:
  // CHECK-NEXT: {{%.*}} = load{{.*}}[[RETVAL]],
  // CHECK-NEXT: ret
  struct my_complex_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define hidden i64 @"\01+[Root classGetComplex]"(
+ (struct my_complex_struct)classGetComplex __attribute__((objc_direct)) {
  struct my_complex_struct st = {.a = 42};
  return st;
  // CHECK: ret i64
}

// CHECK-LABEL: define hidden void @"\01-[Root getAggregate]"(
- (struct my_aggregate_struct)getAggregate __attribute__((objc_direct)) {
  // CHECK: ptr dead_on_unwind noalias writable sret(%struct.my_aggregate_struct) align 4 [[RETVAL:%[^,]*]],

  // loading parameters
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
  // CHECK: br label

  // return
  // CHECK-LABEL: return:
  // CHECK: ret void
  struct my_aggregate_struct st = {.a = 42};
  return st;
}

// CHECK-LABEL: define hidden void @"\01+[Root classGetAggregate]"(
+ (struct my_aggregate_struct)classGetAggregate __attribute__((objc_direct)) {
  struct my_aggregate_struct st = {.a = 42};
  return st;
  // CHECK: ret void
}

// CHECK-LABEL: define hidden void @"\01-[Root accessCmd]"(
- (void)accessCmd __attribute__((objc_direct)) {
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca ptr,
  // CHECK-NEXT: [[CMDVAL:%_cmd]] = alloca ptr,

  // loading the _cmd selector
  // CHECK-LABEL: objc_direct_method.cont:
  // CHECK-NEXT: [[CMD1:%.*]] = load ptr, ptr @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: store ptr [[CMD1]], ptr [[CMDVAL]],
  SEL sel = _cmd;
}

@end
// CHECK-LABEL: define hidden i32 @"\01-[Root intProperty]"

// Check the synthesized objectProperty calls objc_getProperty(); this also
// checks that the synthesized method passes undef for the `cmd` argument.
// CHECK-LABEL: define hidden ptr @"\01-[Root objectProperty]"(
// CHECK-LABEL: objc_direct_method.cont:
// CHECK-NEXT: [[SELFVAL:%.*]] = load {{.*}} %self.addr,
// CHECK-NEXT: [[IVAR:%.*]] = load {{.*}} @"OBJC_IVAR_$_Root._objectProperty",
// CHECK-NEXT: call ptr @objc_getProperty(ptr noundef [[SELFVAL]], ptr noundef poison, i64 noundef [[IVAR]], {{.*}})

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
// CHECK-LABEL: define hidden i32 @"\01-[Foo directMethodInExtension]"(
- (int)directMethodInExtension {
  return 42;
}
// CHECK-LABEL: define hidden i32 @"\01-[Foo getDirect_setDynamic]"(
// CHECK-LABEL: define internal void @"\01-[Foo setGetDirect_setDynamic:]"(
// CHECK-LABEL: define internal i32 @"\01-[Foo getDynamic_setDirect]"(
// CHECK-LABEL: define hidden void @"\01-[Foo setGetDynamic_setDirect:]"(
// CHECK-LABEL: define internal void @"\01-[Foo .cxx_destruct]"(
@end

@implementation Foo (Cat)
// CHECK-LABEL: define hidden i32 @"\01-[Foo directMethodInCategory]"(
- (int)directMethodInCategory {
  return 42;
}
// CHECK-LABEL: define hidden i32 @"\01-[Foo directMethodInCategoryNoDecl]"(
- (int)directMethodInCategoryNoDecl __attribute__((objc_direct)) {
  return 42;
}
@end

int useRoot(Root *r) {
  // CHECK-LABEL: define{{.*}} i32 @useRoot
  // CHECK: %{{[^ ]*}} = call i32  @"\01-[Root getInt]"
  // CHECK: %{{[^ ]*}} = call i32  @"\01-[Root intProperty]"
  // CHECK: %{{[^ ]*}} = call i32  @"\01-[Root intProperty2]"
  return [r getInt] + [r intProperty] + [r intProperty2];
}

int useFoo(Foo *f) {
  // CHECK-LABEL: define{{.*}} i32 @useFoo
  // CHECK: call void @"\01-[Foo setGetDynamic_setDirect:]"
  // CHECK: %{{[^ ]*}} = call i32 @"\01-[Foo getDirect_setDynamic]"
  // CHECK: %{{[^ ]*}} = call i32 @"\01-[Foo directMethodInExtension]"
  // CHECK: %{{[^ ]*}} = call i32 @"\01-[Foo directMethodInCategory]"
  // CHECK: %{{[^ ]*}} = call i32 @"\01-[Foo directMethodInCategoryNoDecl]"
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

int useRootDeclOnly(RootDeclOnly *r) {
  // CHECK-LABEL: define{{.*}} i32 @useRootDeclOnly
  // CHECK: %{{[^ ]*}} = call i32 @"\01-[RootDeclOnly intProperty]"
  return [r intProperty];
}
