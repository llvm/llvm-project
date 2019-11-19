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
@end

@implementation Root
// CHECK-LABEL: define hidden i32 @"\01-[Root getInt]"(
- (int)getInt __attribute__((objc_direct)) {
  // loading parameters
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[RETVAL:%.*]] = alloca
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca %0*,
  // CHECK-NEXT: [[_CMDADDR:%.*]] = alloca i8*,
  // CHECK-NEXT: store %0* %{{.*}}, %0** [[SELFADDR]],
  // CHECK-NEXT: store i8* %{{.*}}, i8** [[_CMDADDR]],

  // self nil-check
  // CHECK-NEXT: [[SELF:%.*]] = load %0*, %0** [[SELFADDR]],
  // CHECK-NEXT: [[NILCHECK:%.*]] = icmp eq %0* [[SELF]], null
  // CHECK-NEXT: br i1 [[NILCHECK]],

  // setting return value to nil
  // CHECK-LABEL: objc_direct_method.self_is_nil:
  // CHECK: [[RET0:%.*]] = bitcast{{.*}}[[RETVAL]]
  // CHECK-NEXT: call void @llvm.memset{{[^(]*}}({{[^,]*}}[[RET0]], i8 0,
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
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca i8*,
  // CHECK-NEXT: [[_CMDADDR:%.*]] = alloca i8*,
  // CHECK-NEXT: store i8* %{{.*}}, i8** [[SELFADDR]],
  // CHECK-NEXT: store i8* %{{.*}}, i8** [[_CMDADDR]],

  // [self self]
  // CHECK-NEXT: [[SELF:%.*]] = load i8*, i8** [[SELFADDR]],
  // CHECK-NEXT: [[SELFSEL:%.*]] = load i8*, i8** @OBJC_SELECTOR_REFERENCES_
  // CHECK-NEXT: [[SELF0:%.*]] = call {{.*}} @objc_msgSend
  // CHECK-NEXT: store i8* [[SELF0]], i8** [[SELFADDR]],

  // return
  // CHECK-NEXT: ret
  return 42;
}

// CHECK-LABEL: define hidden i64 @"\01-[Root getComplex]"(
- (struct my_complex_struct)getComplex __attribute__((objc_direct)) {
  // loading parameters
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[RETVAL:%.*]] = alloca
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca %0*,
  // CHECK-NEXT: [[_CMDADDR:%.*]] = alloca i8*,
  // CHECK-NEXT: store %0* %{{.*}}, %0** [[SELFADDR]],
  // CHECK-NEXT: store i8* %{{.*}}, i8** [[_CMDADDR]],

  // self nil-check
  // CHECK-NEXT: [[SELF:%.*]] = load %0*, %0** [[SELFADDR]],
  // CHECK-NEXT: [[NILCHECK:%.*]] = icmp eq %0* [[SELF]], null
  // CHECK-NEXT: br i1 [[NILCHECK]],

  // setting return value to nil
  // CHECK-LABEL: objc_direct_method.self_is_nil:
  // CHECK: [[RET0:%.*]] = bitcast{{.*}}[[RETVAL]]
  // CHECK-NEXT: call void @llvm.memset{{[^(]*}}({{[^,]*}}[[RET0]], i8 0,
  // CHECK-NEXT: br label

  // set value
  // CHECK-LABEL: objc_direct_method.cont:
  // CHECK: [[RET1:%.*]] = bitcast{{.*}}[[RETVAL]]
  // CHECK-NEXT: call void @llvm.memcpy{{[^(]*}}({{[^,]*}}[[RET1]],
  // CHECK-NEXT: br label

  // return
  // CHECK-LABEL: return:
  // CHECK: [[RET2:%.*]] = bitcast{{.*}}[[RETVAL]]
  // CHECK-NEXT: {{%.*}} = load{{.*}}[[RET2]],
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
  // CHECK: %struct.my_aggregate_struct* noalias sret [[RETVAL:%[^,]*]],

  // loading parameters
  // CHECK-LABEL: entry:
  // CHECK-NEXT: [[SELFADDR:%.*]] = alloca %0*,
  // CHECK-NEXT: [[_CMDADDR:%.*]] = alloca i8*,
  // CHECK-NEXT: store %0* %{{.*}}, %0** [[SELFADDR]],
  // CHECK-NEXT: store i8* %{{.*}}, i8** [[_CMDADDR]],

  // self nil-check
  // CHECK-NEXT: [[SELF:%.*]] = load %0*, %0** [[SELFADDR]],
  // CHECK-NEXT: [[NILCHECK:%.*]] = icmp eq %0* [[SELF]], null
  // CHECK-NEXT: br i1 [[NILCHECK]],

  // setting return value to nil
  // CHECK-LABEL: objc_direct_method.self_is_nil:
  // CHECK: [[RET0:%.*]] = bitcast{{.*}}[[RETVAL]]
  // CHECK-NEXT: call void @llvm.memset{{[^(]*}}({{[^,]*}}[[RET0]], i8 0,
  // CHECK-NEXT: br label

  // set value
  // CHECK-LABEL: objc_direct_method.cont:
  // CHECK: [[RET1:%.*]] = bitcast{{.*}}[[RETVAL]]
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

@end

@interface Foo : Root {
  id __strong _cause_cxx_destruct;
}
@property(nonatomic, readonly, direct) int getDirect_setDynamic;
@property(nonatomic, readonly) int getDynamic_setDirect;
@end

@interface Foo ()
@property(nonatomic, readwrite) int getDirect_setDynamic;
@property(nonatomic, readwrite, direct) int getDynamic_setDirect;
@end

__attribute__((objc_direct_members))
@implementation Foo
// CHECK-LABEL: define hidden i32 @"\01-[Foo getDirect_setDynamic]"(
// CHECK-LABEL: define internal void @"\01-[Foo setGetDirect_setDynamic:]"(
// CHECK-LABEL: define internal i32 @"\01-[Foo getDynamic_setDirect]"(
// CHECK-LABEL: define hidden void @"\01-[Foo setGetDynamic_setDirect:]"(
// CHECK-LABEL: define internal void @"\01-[Foo .cxx_destruct]"(
@end
