// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -O0 %s -o - 2>&1 | FileCheck %s

typedef unsigned long size_t;

struct Foo {
  int t[10];
};

#define PS(N) __attribute__((pass_object_size(N)))

int gi = 0;

// CHECK-LABEL: define i32 @ObjectSize0(i8* %{{.*}}, i64)
int ObjectSize0(void *const p PS(0)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 0);
}

// CHECK-LABEL: define i32 @ObjectSize1(i8* %{{.*}}, i64)
int ObjectSize1(void *const p PS(1)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 1);
}

// CHECK-LABEL: define i32 @ObjectSize2(i8* %{{.*}}, i64)
int ObjectSize2(void *const p PS(2)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 2);
}

// CHECK-LABEL: define i32 @ObjectSize3(i8* %{{.*}}, i64)
int ObjectSize3(void *const p PS(3)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 3);
}

// CHECK-LABEL: define void @test1
void test1() {
  struct Foo t[10];

  // CHECK: call i32 @ObjectSize0(i8* %{{.*}}, i64 360)
  gi = ObjectSize0(&t[1]);
  // CHECK: call i32 @ObjectSize1(i8* %{{.*}}, i64 360)
  gi = ObjectSize1(&t[1]);
  // CHECK: call i32 @ObjectSize2(i8* %{{.*}}, i64 360)
  gi = ObjectSize2(&t[1]);
  // CHECK: call i32 @ObjectSize3(i8* %{{.*}}, i64 360)
  gi = ObjectSize3(&t[1]);

  // CHECK: call i32 @ObjectSize0(i8* %{{.*}}, i64 356)
  gi = ObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize1(i8* %{{.*}}, i64 36)
  gi = ObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize2(i8* %{{.*}}, i64 356)
  gi = ObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize3(i8* %{{.*}}, i64 36)
  gi = ObjectSize3(&t[1].t[1]);
}

// CHECK-LABEL: define void @test2
void test2(struct Foo *t) {
  // CHECK: call i32 @ObjectSize1(i8* %{{.*}}, i64 36)
  gi = ObjectSize1(&t->t[1]);
  // CHECK: call i32 @ObjectSize3(i8* %{{.*}}, i64 36)
  gi = ObjectSize3(&t->t[1]);
}

// CHECK-LABEL: define i32 @_Z27NoViableOverloadObjectSize0Pv
int NoViableOverloadObjectSize0(void *const p) __attribute__((overloadable)) {
  // CHECK: @llvm.objectsize
  return __builtin_object_size(p, 0);
}

// CHECK-LABEL: define i32 @_Z27NoViableOverloadObjectSize1Pv
int NoViableOverloadObjectSize1(void *const p) __attribute__((overloadable)) {
  // CHECK: @llvm.objectsize
  return __builtin_object_size(p, 1);
}

// CHECK-LABEL: define i32 @_Z27NoViableOverloadObjectSize2Pv
int NoViableOverloadObjectSize2(void *const p) __attribute__((overloadable)) {
  // CHECK: @llvm.objectsize
  return __builtin_object_size(p, 2);
}

// CHECK-LABEL: define i32 @_Z27NoViableOverloadObjectSize3Pv
int NoViableOverloadObjectSize3(void *const p) __attribute__((overloadable)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 3);
}

// CHECK-LABEL: define i32 @_Z27NoViableOverloadObjectSize0Pv
// CHECK-NOT: @llvm.objectsize
int NoViableOverloadObjectSize0(void *const p PS(0))
    __attribute__((overloadable)) {
  return __builtin_object_size(p, 0);
}

int NoViableOverloadObjectSize1(void *const p PS(1))
    __attribute__((overloadable)) {
  return __builtin_object_size(p, 1);
}

int NoViableOverloadObjectSize2(void *const p PS(2))
    __attribute__((overloadable)) {
  return __builtin_object_size(p, 2);
}

int NoViableOverloadObjectSize3(void *const p PS(3))
    __attribute__((overloadable)) {
  return __builtin_object_size(p, 3);
}

const static int SHOULDNT_BE_CALLED = -100;
int NoViableOverloadObjectSize0(void *const p PS(0))
    __attribute__((overloadable, enable_if(p == 0, "never selected"))) {
  return SHOULDNT_BE_CALLED;
}

int NoViableOverloadObjectSize1(void *const p PS(1))
    __attribute__((overloadable, enable_if(p == 0, "never selected"))) {
  return SHOULDNT_BE_CALLED;
}

int NoViableOverloadObjectSize2(void *const p PS(2))
    __attribute__((overloadable, enable_if(p == 0, "never selected"))) {
  return SHOULDNT_BE_CALLED;
}

int NoViableOverloadObjectSize3(void *const p PS(3))
    __attribute__((overloadable, enable_if(p == 0, "never selected"))) {
  return SHOULDNT_BE_CALLED;
}

// CHECK-LABEL: define void @test3
void test3() {
  struct Foo t[10];

  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* %{{.*}}, i64 360)
  gi = NoViableOverloadObjectSize0(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* %{{.*}}, i64 360)
  gi = NoViableOverloadObjectSize1(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* %{{.*}}, i64 360)
  gi = NoViableOverloadObjectSize2(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* %{{.*}}, i64 360)
  gi = NoViableOverloadObjectSize3(&t[1]);

  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* %{{.*}}, i64 356)
  gi = NoViableOverloadObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* %{{.*}}, i64 36)
  gi = NoViableOverloadObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* %{{.*}}, i64 356)
  gi = NoViableOverloadObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* %{{.*}}, i64 36)
  gi = NoViableOverloadObjectSize3(&t[1].t[1]);
}

// CHECK-LABEL: define void @test4
void test4(struct Foo *t) {
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* %{{.*}}, i64 %{{.*}})
  gi = NoViableOverloadObjectSize0(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* %{{.*}}, i64 %{{.*}})
  gi = NoViableOverloadObjectSize1(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* %{{.*}}, i64 %{{.*}})
  gi = NoViableOverloadObjectSize2(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* %{{.*}}, i64 0)
  gi = NoViableOverloadObjectSize3(&t[1]);

  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* %{{.*}}, i64 %{{.*}})
  gi = NoViableOverloadObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* %{{.*}}, i64 36)
  gi = NoViableOverloadObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* %{{.*}}, i64 %{{.*}})
  gi = NoViableOverloadObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* %{{.*}}, i64 36)
  gi = NoViableOverloadObjectSize3(&t[1].t[1]);
}

void test5() {
  struct Foo t[10];

  int (*f)(void *) = &NoViableOverloadObjectSize0;
  gi = f(&t[1]);
}

// CHECK-LABEL: define i32 @IndirectObjectSize0
int IndirectObjectSize0(void *const p PS(0)) {
  // CHECK: call i32 @ObjectSize0(i8* %{{.*}}, i64 %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize0(p);
}

// CHECK-LABEL: define i32 @IndirectObjectSize1
int IndirectObjectSize1(void *const p PS(1)) {
  // CHECK: call i32 @ObjectSize1(i8* %{{.*}}, i64 %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize1(p);
}

// CHECK-LABEL: define i32 @IndirectObjectSize2
int IndirectObjectSize2(void *const p PS(2)) {
  // CHECK: call i32 @ObjectSize2(i8* %{{.*}}, i64 %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize2(p);
}

// CHECK-LABEL: define i32 @IndirectObjectSize3
int IndirectObjectSize3(void *const p PS(3)) {
  // CHECK: call i32 @ObjectSize3(i8* %{{.*}}, i64 %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize3(p);
}

int Overload0(void *, size_t, void *, size_t);
int OverloadNoSize(void *, void *);

int OverloadedObjectSize(void *const p PS(0),
                         void *const c PS(0))
    __attribute__((overloadable)) __asm__("Overload0");

int OverloadedObjectSize(void *const p, void *const c)
    __attribute__((overloadable)) __asm__("OverloadNoSize");

// CHECK-LABEL: define void @test6
void test6() {
  int known[10], *opaque;

  // CHECK: call i32 @"\01Overload0"
  gi = OverloadedObjectSize(&known[0], &known[0]);

  // CHECK: call i32 @"\01Overload0"
  gi = OverloadedObjectSize(&known[0], opaque);

  // CHECK: call i32 @"\01Overload0"
  gi = OverloadedObjectSize(opaque, &known[0]);

  // CHECK: call i32 @"\01Overload0"
  gi = OverloadedObjectSize(opaque, opaque);
}

int Identity(void *p, size_t i) { return i; }

// CHECK-NOT: define void @AsmObjectSize
int AsmObjectSize0(void *const p PS(0)) __asm__("Identity");

int AsmObjectSize1(void *const p PS(1)) __asm__("Identity");

int AsmObjectSize2(void *const p PS(2)) __asm__("Identity");

int AsmObjectSize3(void *const p PS(3)) __asm__("Identity");

// CHECK-LABEL: define void @test7
void test7() {
  struct Foo t[10];

  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 360)
  gi = AsmObjectSize0(&t[1]);
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 360)
  gi = AsmObjectSize1(&t[1]);
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 360)
  gi = AsmObjectSize2(&t[1]);
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 360)
  gi = AsmObjectSize3(&t[1]);

  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 356)
  gi = AsmObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 36)
  gi = AsmObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 356)
  gi = AsmObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 36)
  gi = AsmObjectSize3(&t[1].t[1]);
}

// CHECK-LABEL: define void @test8
void test8(struct Foo *t) {
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 36)
  gi = AsmObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* %{{.*}}, i64 36)
  gi = AsmObjectSize3(&t[1].t[1]);
}

void DifferingObjectSize0(void *const p __attribute__((pass_object_size(0))));
void DifferingObjectSize1(void *const p __attribute__((pass_object_size(1))));
void DifferingObjectSize2(void *const p __attribute__((pass_object_size(2))));
void DifferingObjectSize3(void *const p __attribute__((pass_object_size(3))));

// CHECK-LABEL: define void @test9
void test9(void *const p __attribute__((pass_object_size(0)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize2(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize0(p);
  DifferingObjectSize1(p);

  // CHECK: call void @DifferingObjectSize3(i8* %{{.*}}, i64 0)
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define void @test10
void test10(void *const p __attribute__((pass_object_size(1)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize2(p);
  // CHECK: @llvm.objectsize
  DifferingObjectSize0(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize1(p);

  // CHECK: call void @DifferingObjectSize3(i8* %{{.*}}, i64 0)
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define void @test11
void test11(void *const p __attribute__((pass_object_size(2)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize0(p);
  // CHECK: @llvm.objectsize
  DifferingObjectSize1(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize2(p);

  // CHECK: call void @DifferingObjectSize3(i8* %{{.*}}, i64 0)
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define void @test12
void test12(void *const p __attribute__((pass_object_size(3)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize0(p);
  // CHECK: @llvm.objectsize
  DifferingObjectSize1(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize2(p);
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define void @test13
void test13() {
  // Ensuring that we don't lower objectsize if the expression has side-effects
  char c[10];
  char *p = c;

  // CHECK: @llvm.objectsize
  ObjectSize0(p);

  // CHECK-NOT: @llvm.objectsize
  ObjectSize0(++p);
  ObjectSize0(p++);
}
