// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin -emit-llvm -O0 %s -o - 2>&1 | FileCheck %s

typedef unsigned long size_t;

struct Foo {
  int t[10];
};

#define PS(N) __attribute__((pass_object_size(N)))
#define PDS(N) __attribute__((pass_dynamic_object_size(N)))

int gi = 0;

// CHECK-LABEL: define{{.*}} i32 @ObjectSize0(i8* noundef %{{.*}}, i64 noundef %0)
int ObjectSize0(void *const p PS(0)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 0);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize0(i8* noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize0(void *const p PDS(0)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_dynamic_object_size(p, 0);
}

// CHECK-LABEL: define{{.*}} i32 @ObjectSize1(i8* noundef %{{.*}}, i64 noundef %0)
int ObjectSize1(void *const p PS(1)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 1);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize1(i8* noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize1(void *const p PDS(1)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_dynamic_object_size(p, 1);
}

// CHECK-LABEL: define{{.*}} i32 @ObjectSize2(i8* noundef %{{.*}}, i64 noundef %0)
int ObjectSize2(void *const p PS(2)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 2);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize2(i8* noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize2(void *const p PDS(2)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 2);
}

// CHECK-LABEL: define{{.*}} i32 @ObjectSize3(i8* noundef %{{.*}}, i64 noundef %0)
int ObjectSize3(void *const p PS(3)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 3);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize3(i8* noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize3(void *const p PDS(3)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 3);
}

void *malloc(unsigned long) __attribute__((alloc_size(1)));

// CHECK-LABEL: define{{.*}} void @test1
void test1(unsigned long sz) {
  struct Foo t[10];

  // CHECK: call i32 @ObjectSize0(i8* noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize0(&t[1]);
  // CHECK: call i32 @ObjectSize1(i8* noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize1(&t[1]);
  // CHECK: call i32 @ObjectSize2(i8* noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize2(&t[1]);
  // CHECK: call i32 @ObjectSize3(i8* noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize3(&t[1]);

  // CHECK: call i32 @ObjectSize0(i8* noundef %{{.*}}, i64 noundef 356)
  gi = ObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize1(i8* noundef %{{.*}}, i64 noundef 36)
  gi = ObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize2(i8* noundef %{{.*}}, i64 noundef 356)
  gi = ObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize3(i8* noundef %{{.*}}, i64 noundef 36)
  gi = ObjectSize3(&t[1].t[1]);

  char *ptr = (char *)malloc(sz);

  // CHECK: [[REG:%.*]] = call i64 @llvm.objectsize.i64.p0i8({{.*}}, i1 false, i1 true, i1 true)
  // CHECK: call i32 @DynamicObjectSize0(i8* noundef %{{.*}}, i64 noundef [[REG]])
  gi = DynamicObjectSize0(ptr);

  // CHECK: [[WITH_OFFSET:%.*]] = getelementptr
  // CHECK: [[REG:%.*]] = call i64 @llvm.objectsize.i64.p0i8(i8* [[WITH_OFFSET]], i1 false, i1 true, i1 true)
  // CHECK: call i32 @DynamicObjectSize0(i8* noundef {{.*}}, i64 noundef [[REG]])
  gi = DynamicObjectSize0(ptr+10);

  // CHECK: [[REG:%.*]] = call i64 @llvm.objectsize.i64.p0i8({{.*}}, i1 true, i1 true, i1 true)
  // CHECK: call i32 @DynamicObjectSize2(i8* noundef {{.*}}, i64 noundef [[REG]])
  gi = DynamicObjectSize2(ptr);
}

// CHECK-LABEL: define{{.*}} void @test2
void test2(struct Foo *t) {
  // CHECK: [[VAR:%[0-9]+]] = call i64 @llvm.objectsize
  // CHECK: call i32 @ObjectSize1(i8* noundef %{{.*}}, i64 noundef [[VAR]])
  gi = ObjectSize1(&t->t[1]);
  // CHECK: call i32 @ObjectSize3(i8* noundef %{{.*}}, i64 noundef 36)
  gi = ObjectSize3(&t->t[1]);
}

// CHECK-LABEL: define{{.*}} i32 @_Z27NoViableOverloadObjectSize0Pv
int NoViableOverloadObjectSize0(void *const p) __attribute__((overloadable)) {
  // CHECK: @llvm.objectsize
  return __builtin_object_size(p, 0);
}

// CHECK-LABEL: define{{.*}} i32 @_Z34NoViableOverloadDynamicObjectSize0Pv
int NoViableOverloadDynamicObjectSize0(void *const p)
  __attribute__((overloadable)) {
  // CHECK: @llvm.objectsize
  return __builtin_object_size(p, 0);
}

// CHECK-LABEL: define{{.*}} i32 @_Z27NoViableOverloadObjectSize1Pv
int NoViableOverloadObjectSize1(void *const p) __attribute__((overloadable)) {
  // CHECK: @llvm.objectsize
  return __builtin_object_size(p, 1);
}

// CHECK-LABEL: define{{.*}} i32 @_Z27NoViableOverloadObjectSize2Pv
int NoViableOverloadObjectSize2(void *const p) __attribute__((overloadable)) {
  // CHECK: @llvm.objectsize
  return __builtin_object_size(p, 2);
}

// CHECK-LABEL: define{{.*}} i32 @_Z27NoViableOverloadObjectSize3Pv
int NoViableOverloadObjectSize3(void *const p) __attribute__((overloadable)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 3);
}

// CHECK-LABEL: define{{.*}} i32 @_Z27NoViableOverloadObjectSize0Pv
// CHECK-NOT: @llvm.objectsize
int NoViableOverloadObjectSize0(void *const p PS(0))
    __attribute__((overloadable)) {
  return __builtin_object_size(p, 0);
}

int NoViableOverloadDynamicObjectSize0(void *const p PDS(0))
  __attribute__((overloadable)) {
  return __builtin_dynamic_object_size(p, 0);
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

// CHECK-LABEL: define{{.*}} void @test3
void test3(void) {
  struct Foo t[10];

  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* noundef %{{.*}}, i64 noundef 360)
  gi = NoViableOverloadObjectSize0(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* noundef %{{.*}}, i64 noundef 360)
  gi = NoViableOverloadObjectSize1(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* noundef %{{.*}}, i64 noundef 360)
  gi = NoViableOverloadObjectSize2(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* noundef %{{.*}}, i64 noundef 360)
  gi = NoViableOverloadObjectSize3(&t[1]);

  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* noundef %{{.*}}, i64 noundef 356)
  gi = NoViableOverloadObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* noundef %{{.*}}, i64 noundef 36)
  gi = NoViableOverloadObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* noundef %{{.*}}, i64 noundef 356)
  gi = NoViableOverloadObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* noundef %{{.*}}, i64 noundef 36)
  gi = NoViableOverloadObjectSize3(&t[1].t[1]);

  // CHECK: call i32 @_Z34NoViableOverloadDynamicObjectSize0PvU25pass_dynamic_object_size0(i8* noundef %{{.*}}, i64 noundef 360)
  gi = NoViableOverloadDynamicObjectSize0(&t[1]);
}

// CHECK-LABEL: define{{.*}} void @test4
void test4(struct Foo *t) {
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  gi = NoViableOverloadObjectSize0(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  gi = NoViableOverloadObjectSize1(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  gi = NoViableOverloadObjectSize2(&t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* noundef %{{.*}}, i64 noundef 0)
  gi = NoViableOverloadObjectSize3(&t[1]);

  // CHECK: call i32 @_Z27NoViableOverloadObjectSize0PvU17pass_object_size0(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  gi = NoViableOverloadObjectSize0(&t[1].t[1]);
  // CHECK: [[VAR:%[0-9]+]] = call i64 @llvm.objectsize
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize1PvU17pass_object_size1(i8* noundef %{{.*}}, i64 noundef [[VAR]])
  gi = NoViableOverloadObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize2PvU17pass_object_size2(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  gi = NoViableOverloadObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @_Z27NoViableOverloadObjectSize3PvU17pass_object_size3(i8* noundef %{{.*}}, i64 noundef 36)
  gi = NoViableOverloadObjectSize3(&t[1].t[1]);
}

void test5(void) {
  struct Foo t[10];

  int (*f)(void *) = &NoViableOverloadObjectSize0;
  gi = f(&t[1]);

  int (*g)(void *) = &NoViableOverloadDynamicObjectSize0;
  gi = g(&t[1]);
}

// CHECK-LABEL: define{{.*}} i32 @IndirectObjectSize0
int IndirectObjectSize0(void *const p PS(0)) {
  // CHECK: call i32 @ObjectSize0(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize0(p);
}

// CHECK-LABEL: define{{.*}} i32 @IndirectObjectSize1
int IndirectObjectSize1(void *const p PS(1)) {
  // CHECK: call i32 @ObjectSize1(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize1(p);
}

// CHECK-LABEL: define{{.*}} i32 @IndirectObjectSize2
int IndirectObjectSize2(void *const p PS(2)) {
  // CHECK: call i32 @ObjectSize2(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize2(p);
}

// CHECK-LABEL: define{{.*}} i32 @IndirectObjectSize3
int IndirectObjectSize3(void *const p PS(3)) {
  // CHECK: call i32 @ObjectSize3(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize3(p);
}

int IndirectDynamicObjectSize0(void *const p PDS(0)) {
  // CHECK: call i32 @ObjectSize0(i8* noundef %{{.*}}, i64 noundef %{{.*}})
  // CHECK-NOT: @llvm.objectsize
  return ObjectSize0(p);
}

int Overload0(void *, size_t, void *, size_t);
int OverloadNoSize(void *, void *);

int OverloadedObjectSize(void *const p PS(0),
                         void *const c PS(0))
    __attribute__((overloadable)) __asm__("Overload0");

int OverloadedObjectSize(void *const p, void *const c)
    __attribute__((overloadable)) __asm__("OverloadNoSize");

// CHECK-LABEL: define{{.*}} void @test6
void test6(void) {
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

// CHECK-NOT: define{{.*}} void @AsmObjectSize
int AsmObjectSize0(void *const p PS(0)) __asm__("Identity");

int AsmObjectSize1(void *const p PS(1)) __asm__("Identity");

int AsmObjectSize2(void *const p PS(2)) __asm__("Identity");

int AsmObjectSize3(void *const p PS(3)) __asm__("Identity");

// CHECK-LABEL: define{{.*}} void @test7
void test7(void) {
  struct Foo t[10];

  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 360)
  gi = AsmObjectSize0(&t[1]);
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 360)
  gi = AsmObjectSize1(&t[1]);
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 360)
  gi = AsmObjectSize2(&t[1]);
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 360)
  gi = AsmObjectSize3(&t[1]);

  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 356)
  gi = AsmObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 36)
  gi = AsmObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 356)
  gi = AsmObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 36)
  gi = AsmObjectSize3(&t[1].t[1]);
}

// CHECK-LABEL: define{{.*}} void @test8
void test8(struct Foo *t) {
  // CHECK: [[VAR:%[0-9]+]] = call i64 @llvm.objectsize
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef [[VAR]])
  gi = AsmObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @"\01Identity"(i8* noundef %{{.*}}, i64 noundef 36)
  gi = AsmObjectSize3(&t[1].t[1]);
}

void DifferingObjectSize0(void *const p __attribute__((pass_object_size(0))));
void DifferingObjectSize1(void *const p __attribute__((pass_object_size(1))));
void DifferingObjectSize2(void *const p __attribute__((pass_object_size(2))));
void DifferingObjectSize3(void *const p __attribute__((pass_object_size(3))));

// CHECK-LABEL: define{{.*}} void @test9
void test9(void *const p __attribute__((pass_object_size(0)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize2(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize0(p);
  DifferingObjectSize1(p);

  // CHECK: call void @DifferingObjectSize3(i8* noundef %{{.*}}, i64 noundef 0)
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define{{.*}} void @test10
void test10(void *const p __attribute__((pass_object_size(1)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize2(p);
  // CHECK: @llvm.objectsize
  DifferingObjectSize0(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize1(p);

  // CHECK: call void @DifferingObjectSize3(i8* noundef %{{.*}}, i64 noundef 0)
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define{{.*}} void @test11
void test11(void *const p __attribute__((pass_object_size(2)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize0(p);
  // CHECK: @llvm.objectsize
  DifferingObjectSize1(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize2(p);

  // CHECK: call void @DifferingObjectSize3(i8* noundef %{{.*}}, i64 noundef 0)
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define{{.*}} void @test12
void test12(void *const p __attribute__((pass_object_size(3)))) {
  // CHECK: @llvm.objectsize
  DifferingObjectSize0(p);
  // CHECK: @llvm.objectsize
  DifferingObjectSize1(p);

  // CHECK-NOT: @llvm.objectsize
  DifferingObjectSize2(p);
  DifferingObjectSize3(p);
}

// CHECK-LABEL: define{{.*}} void @test13
void test13(void) {
  char c[10];
  unsigned i = 0;
  char *p = c;

  // CHECK: @llvm.objectsize
  ObjectSize0(p);

  // Allow side-effects, since they always need to happen anyway. Just make sure
  // we don't perform them twice.
  // CHECK: = add
  // CHECK-NOT: = add
  // CHECK: @llvm.objectsize
  // CHECK: call i32 @ObjectSize0
  ObjectSize0(p + ++i);

  // CHECK: = add
  // CHECK: @llvm.objectsize
  // CHECK-NOT: = add
  // CHECK: call i32 @ObjectSize0
  ObjectSize0(p + i++);
}

// There was a bug where variadic functions with pass_object_size would cause
// problems in the form of failed assertions.
void my_sprintf(char *const c __attribute__((pass_object_size(0))), ...) {}

// CHECK-LABEL: define{{.*}} void @test14
void test14(char *c) {
  // CHECK: @llvm.objectsize
  // CHECK: call void (i8*, i64, ...) @my_sprintf
  my_sprintf(c);

  // CHECK: @llvm.objectsize
  // CHECK: call void (i8*, i64, ...) @my_sprintf
  my_sprintf(c, 1, 2, 3);
}

void pass_size_unsigned(unsigned *const PS(0));

// Bug: we weren't lowering to the proper @llvm.objectsize for pointers that
// don't turn into i8*s, which caused crashes.
// CHECK-LABEL: define{{.*}} void @test15
void test15(unsigned *I) {
  // CHECK: @llvm.objectsize.i64.p0i32
  // CHECK: call void @pass_size_unsigned
  pass_size_unsigned(I);
}

void pass_size_as1(__attribute__((address_space(1))) void *const PS(0));

void pass_size_unsigned_as1(
    __attribute__((address_space(1))) unsigned *const PS(0));

// CHECK-LABEL: define{{.*}} void @test16
void test16(__attribute__((address_space(1))) unsigned *I) {
  // CHECK: call i64 @llvm.objectsize.i64.p1i8
  // CHECK: call void @pass_size_as1
  pass_size_as1(I);
  // CHECK: call i64 @llvm.objectsize.i64.p1i32
  // CHECK: call void @pass_size_unsigned_as1
  pass_size_unsigned_as1(I);
}

// This used to cause assertion failures, since we'd try to emit the statement
// expression (and definitions for `a`) twice.
// CHECK-LABEL: define{{.*}} void @test17
void test17(char *C) {
  // Check for 65535 to see if we're emitting this pointer twice.
  // CHECK: 65535
  // CHECK-NOT: 65535
  // CHECK: @llvm.objectsize.i64.p0i8(i8* [[PTR:%[^,]+]],
  // CHECK-NOT: 65535
  // CHECK: call i32 @ObjectSize0(i8* noundef [[PTR]]
  ObjectSize0(C + ({ int a = 65535; a; }));
}

// CHECK-LABEL: define{{.*}} void @test18
void test18(char *const p PDS(0)) {
  // CHECK-NOT: llvm.objectsize
  gi = __builtin_dynamic_object_size(p, 0);
  gi = __builtin_object_size(p, 0);
}
