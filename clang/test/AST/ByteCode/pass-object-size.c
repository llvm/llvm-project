// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1                                         -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

/// This is the part of test/CodeGen/pass-object-size.c we can evaluate.

typedef unsigned long size_t;

struct Foo {
  int t[10];
};

#define PS(N) __attribute__((pass_object_size(N)))
#define PDS(N) __attribute__((pass_dynamic_object_size(N)))

int gi = 0;

// CHECK-LABEL: define{{.*}} i32 @ObjectSize0(ptr noundef %{{.*}}, i64 noundef %0)
int ObjectSize0(void *const p PS(0)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 0);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize0(ptr noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize0(void *const p PDS(0)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_dynamic_object_size(p, 0);
}

// CHECK-LABEL: define{{.*}} i32 @ObjectSize1(ptr noundef %{{.*}}, i64 noundef %0)
int ObjectSize1(void *const p PS(1)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 1);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize1(ptr noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize1(void *const p PDS(1)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_dynamic_object_size(p, 1);
}

// CHECK-LABEL: define{{.*}} i32 @ObjectSize2(ptr noundef %{{.*}}, i64 noundef %0)
int ObjectSize2(void *const p PS(2)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 2);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize2(ptr noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize2(void *const p PDS(2)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 2);
}

// CHECK-LABEL: define{{.*}} i32 @ObjectSize3(ptr noundef %{{.*}}, i64 noundef %0)
int ObjectSize3(void *const p PS(3)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 3);
}

// CHECK-LABEL: define{{.*}} i32 @DynamicObjectSize3(ptr noundef %{{.*}}, i64 noundef %0)
int DynamicObjectSize3(void *const p PDS(3)) {
  // CHECK-NOT: @llvm.objectsize
  return __builtin_object_size(p, 3);
}

void *malloc(unsigned long) __attribute__((alloc_size(1)));

// CHECK-LABEL: define{{.*}} void @test1
void test1(unsigned long sz) {
  struct Foo t[10];

  // CHECK: call i32 @ObjectSize0(ptr noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize0(&t[1]);
  // CHECK: call i32 @ObjectSize1(ptr noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize1(&t[1]);
  // CHECK: call i32 @ObjectSize2(ptr noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize2(&t[1]);
  // CHECK: call i32 @ObjectSize3(ptr noundef %{{.*}}, i64 noundef 360)
  gi = ObjectSize3(&t[1]);

  // CHECK: call i32 @ObjectSize0(ptr noundef %{{.*}}, i64 noundef 356)
  gi = ObjectSize0(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize1(ptr noundef %{{.*}}, i64 noundef 36)
  gi = ObjectSize1(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize2(ptr noundef %{{.*}}, i64 noundef 356)
  gi = ObjectSize2(&t[1].t[1]);
  // CHECK: call i32 @ObjectSize3(ptr noundef %{{.*}}, i64 noundef 36)
  gi = ObjectSize3(&t[1].t[1]);

  char *ptr = (char *)malloc(sz);

  // CHECK: [[REG:%.*]] = call i64 @llvm.objectsize.i64.p0({{.*}}, i1 false, i1 true, i1 true)
  // CHECK: call i32 @DynamicObjectSize0(ptr noundef %{{.*}}, i64 noundef [[REG]])
  gi = DynamicObjectSize0(ptr);

  // CHECK: [[WITH_OFFSET:%.*]] = getelementptr
  // CHECK: [[REG:%.*]] = call i64 @llvm.objectsize.i64.p0(ptr [[WITH_OFFSET]], i1 false, i1 true, i1 true)
  // CHECK: call i32 @DynamicObjectSize0(ptr noundef {{.*}}, i64 noundef [[REG]])
  gi = DynamicObjectSize0(ptr+10);

  // CHECK: [[REG:%.*]] = call i64 @llvm.objectsize.i64.p0({{.*}}, i1 true, i1 true, i1 true)
  // CHECK: call i32 @DynamicObjectSize2(ptr noundef {{.*}}, i64 noundef [[REG]])
  gi = DynamicObjectSize2(ptr);
}
