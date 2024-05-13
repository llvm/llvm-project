// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -fobjc-arc -fobjc-runtime-has-weak -o - %s | FileCheck %s

// CHECK: %[[STRUCT_TRIVIAL:.*]] = type { i32 }
// CHECK: %[[STRUCT_TRIVIALBIG:.*]] = type { [64 x i32] }
// CHECK: %[[STRUCT_STRONG:.*]] = type { ptr }
// CHECK: %[[STRUCT_WEAK:.*]] = type { ptr }

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

// CHECK: define{{.*}} i32 @testTrivial()
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_TRIVIAL]], align 4
// CHECK-NEXT: call void @func0(ptr noundef %[[RETVAL]])
// CHECK-NOT: memcpy
// CHECK: ret i32 %

void func0(Trivial *);

Trivial testTrivial(void) {
  Trivial a;
  func0(&a);
  return a;
}

void func1(TrivialBig *);

// CHECK: define{{.*}} void @testTrivialBig(ptr dead_on_unwind noalias writable sret(%[[STRUCT_TRIVIALBIG]]) align 4 %[[AGG_RESULT:.*]])
// CHECK: call void @func1(ptr noundef %[[AGG_RESULT]])
// CHECK-NEXT: ret void

TrivialBig testTrivialBig(void) {
  TrivialBig a;
  func1(&a);
  return a;
}

// CHECK: define{{.*}} ptr @testStrong()
// CHECK: %[[RETVAL:.*]] = alloca %[[STRUCT_STRONG]], align 8
// CHECK: %[[NRVO:.*]] = alloca i1, align 1
// CHECK: call void @__default_constructor_8_s0(ptr %[[RETVAL]])
// CHECK: store i1 true, ptr %[[NRVO]], align 1
// CHECK: %[[NRVO_VAL:.*]] = load i1, ptr %[[NRVO]], align 1
// CHECK: br i1 %[[NRVO_VAL]],

// CHECK: call void @__destructor_8_s0(ptr %[[RETVAL]])
// CHECK: br

// CHECK: %[[COERCE_DIVE:.*]] = getelementptr inbounds %[[STRUCT_STRONG]], ptr %[[RETVAL]], i32 0, i32 0
// CHECK: %[[V2:.*]] = load ptr, ptr %[[COERCE_DIVE]], align 8
// CHECK: ret ptr %[[V2]]

Strong testStrong(void) {
  Strong a;
  return a;
}

// CHECK: define{{.*}} void @testWeak(ptr dead_on_unwind noalias writable sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_RESULT:.*]])
// CHECK: %[[NRVO:.*]] = alloca i1, align 1
// CHECK: call void @__default_constructor_8_w0(ptr %[[AGG_RESULT]])
// CHECK: store i1 true, ptr %[[NRVO]], align 1
// CHECK: %[[NRVO_VAL:.*]] = load i1, ptr %[[NRVO]], align 1
// CHECK: br i1 %[[NRVO_VAL]],

// CHECK: call void @__destructor_8_w0(ptr %[[AGG_RESULT]])
// CHECK: br

// CHECK-NOT: call
// CHECK: ret void

Weak testWeak(void) {
  Weak a;
  return a;
}

// CHECK: define{{.*}} void @testWeak2(
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

// CHECK: define internal void @"\01-[C1 foo1]"(ptr dead_on_unwind noalias writable sret(%[[STRUCT_WEAK]]) align 8 %[[AGG_RESULT:.*]], ptr noundef %{{.*}}, ptr noundef %{{.*}})
// CHECK: %[[NRVO:.*]] = alloca i1, align 1
// CHECK: call void @__default_constructor_8_w0(ptr %[[AGG_RESULT]])
// CHECK: store i1 true, ptr %[[NRVO]], align 1
// CHECK: %[[NRVO_VAL:.*]] = load i1, ptr %[[NRVO]], align 1
// CHECK: br i1 %[[NRVO_VAL]],

// CHECK: call void @__destructor_8_w0(ptr %[[AGG_RESULT]])
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
