// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck %s

// CHECK: %[[STRUCT_TRIVIAL:.*]] = type { i32 }
// CHECK: %[[STRUCT_TRIVIALBIG:.*]] = type { [64 x i32] }
// CHECK: %[[STRUCT_STRONG:.*]] = type { i8* }
// CHECK: %[[STRUCT_WEAK:.*]] = type { i8* }

typedef struct {
  int x;
} Trivial;

typedef struct {
  int x[64];
} TrivialBig;

typedef struct {
  id x;
} Strong;

typedef struct {
  __weak id x;
} Weak;

// CHECK: define i32 @testTrivial()
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_TRIVIAL]], align 4
// CHECK-NEXT: call void @func0(%[[STRUCT_TRIVIAL]]* %[[RETVAL]])
// CHECK-NOT: memcpy
// CHECK: ret i32 %

void func0(Trivial *);

Trivial testTrivial(void) {
  Trivial a;
  func0(&a);
  return a;
}

void func1(TrivialBig *);

// CHECK: define void @testTrivialBig(%[[STRUCT_TRIVIALBIG]]* noalias sret(%[[STRUCT_TRIVIALBIG]]) align 4 %[[AGG_RESULT:.*]])
// CHECK: call void @func1(%[[STRUCT_TRIVIALBIG]]* %[[AGG_RESULT]])
// CHECK-NEXT: ret void

TrivialBig testTrivialBig(void) {
  TrivialBig a;
  func1(&a);
  return a;
}

// CHECK: define i8* @testStrong()
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[NRVO:.*]] = alloca i1, align 1
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_STRONG]]* %[[RETVAL]] to i8**
// CHECK: call void @__default_constructor_8_s0(i8** %[[V0]])
// CHECK: store i1 true, i1* %[[NRVO]], align 1
// CHECK: %[[NRVO_VAL:.*]] = load i1, i1* %[[NRVO]], align 1
// CHECK: br i1 %[[NRVO_VAL]],

// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_STRONG]]* %[[RETVAL]] to i8**
// CHECK: call void @__destructor_8_s0(i8** %[[V1]])
// CHECK: br

// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_STRONG]], %[[STRUCT_STRONG]]* %[[RETVAL]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load i8*, i8** %[[COERCE_DIVE]], align 8
// CHECK: ret i8* %[[V2]]

Strong testStrong(void) {
  Strong a;
  return a;
}

// CHECK: define void @testWeak(%[[STRUCT_WEAK]]* noalias sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_RESULT:.*]])
// CHECK: %[[NRVO:.*]] = alloca i1, align 1
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_RESULT]] to i8**
// CHECK: call void @__default_constructor_8_w0(i8** %[[V0]])
// CHECK: store i1 true, i1* %[[NRVO]], align 1
// CHECK: %[[NRVO_VAL:.*]] = load i1, i1* %[[NRVO]], align 1
// CHECK: br i1 %[[NRVO_VAL]],

// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_RESULT]] to i8**
// CHECK: call void @__destructor_8_w0(i8** %[[V1]])
// CHECK: br

// CHECK-NOT: call
// CHECK: ret void

Weak testWeak(void) {
  Weak a;
  return a;
}

// CHECK: define void @testWeak2(
// CHECK: call void @__default_constructor_8_w0(
// CHECK: call void @__default_constructor_8_w0(
// CHECK: call void @__copy_constructor_8_8_w0(
// CHECK: call void @__copy_constructor_8_8_w0(
// CHECK: call void @__destructor_8_w0(
// CHECK: call void @__destructor_8_w0(

Weak testWeak2(int c) {
  Weak a, b;
  if (c)
    return a;
  else
    return b;
}

// CHECK: define internal void @"\01-[C1 foo1]"(%[[STRUCT_WEAK]]* noalias sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_RESULT:.*]], %{{.*}}* %{{.*}}, i8* %{{.*}})
// CHECK: %[[NRVO:.*]] = alloca i1, align 1
// CHECK: %[[V0:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_RESULT]] to i8**
// CHECK: call void @__default_constructor_8_w0(i8** %[[V0]])
// CHECK: store i1 true, i1* %[[NRVO]], align 1
// CHECK: %[[NRVO_VAL:.*]] = load i1, i1* %[[NRVO]], align 1
// CHECK: br i1 %[[NRVO_VAL]],

// CHECK: %[[V1:.*]] = bitcast %[[STRUCT_WEAK]]* %[[AGG_RESULT]] to i8**
// CHECK: call void @__destructor_8_w0(i8** %[[V1]])
// CHECK: br

// CHECK-NOT: call
// CHECK: ret void

__attribute__((objc_root_class))
@interface C1
- (Weak)foo1;
@end

@implementation C1
- (Weak)foo1 {
  Weak a;
  return a;
}
@end
