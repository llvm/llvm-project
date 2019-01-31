// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm %s -o - 2>&1 | FileCheck %s

#define NULL ((void *)0)

int gi;

typedef unsigned long size_t;

// CHECK-DAG-RE: define void @my_malloc({{.*}}) #[[MALLOC_ATTR_NUMBER:[0-9]+]]
// N.B. LLVM's allocsize arguments are base-0, whereas ours are base-1 (for
// compat with GCC)
// CHECK-DAG-RE: attributes #[[MALLOC_ATTR_NUMBER]] = {.*allocsize(0).*}
void *my_malloc(size_t) __attribute__((alloc_size(1)));

// CHECK-DAG-RE: define void @my_calloc({{.*}}) #[[CALLOC_ATTR_NUMBER:[0-9]+]]
// CHECK-DAG-RE: attributes #[[CALLOC_ATTR_NUMBER]] = {.*allocsize(0, 1).*}
void *my_calloc(size_t, size_t) __attribute__((alloc_size(1, 2)));

// CHECK-LABEL: @test1
void test1() {
  void *const vp = my_malloc(100);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 0);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 1);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 2);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 3);

  void *const arr = my_calloc(100, 5);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 0);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 1);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 2);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 3);

  // CHECK: store i32 100
  gi = __builtin_object_size(my_malloc(100), 0);
  // CHECK: store i32 100
  gi = __builtin_object_size(my_malloc(100), 1);
  // CHECK: store i32 100
  gi = __builtin_object_size(my_malloc(100), 2);
  // CHECK: store i32 100
  gi = __builtin_object_size(my_malloc(100), 3);

  // CHECK: store i32 500
  gi = __builtin_object_size(my_calloc(100, 5), 0);
  // CHECK: store i32 500
  gi = __builtin_object_size(my_calloc(100, 5), 1);
  // CHECK: store i32 500
  gi = __builtin_object_size(my_calloc(100, 5), 2);
  // CHECK: store i32 500
  gi = __builtin_object_size(my_calloc(100, 5), 3);

  void *const zeroPtr = my_malloc(0);
  // CHECK: store i32 0
  gi = __builtin_object_size(zeroPtr, 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(my_malloc(0), 0);

  void *const zeroArr1 = my_calloc(0, 1);
  void *const zeroArr2 = my_calloc(1, 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(zeroArr1, 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(zeroArr2, 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(my_calloc(1, 0), 0);
  // CHECK: store i32 0
  gi = __builtin_object_size(my_calloc(0, 1), 0);
}

// CHECK-LABEL: @test2
void test2() {
  void *const vp = my_malloc(gi);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(vp, 0);

  void *const arr1 = my_calloc(gi, 1);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr1, 0);

  void *const arr2 = my_calloc(1, gi);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr2, 0);
}

// CHECK-LABEL: @test3
void test3() {
  char *const buf = (char *)my_calloc(100, 5);
  // CHECK: store i32 500
  gi = __builtin_object_size(buf, 0);
  // CHECK: store i32 500
  gi = __builtin_object_size(buf, 1);
  // CHECK: store i32 500
  gi = __builtin_object_size(buf, 2);
  // CHECK: store i32 500
  gi = __builtin_object_size(buf, 3);
}

struct Data {
  int a;
  int t[10];
  char pad[3];
  char end[1];
};

// CHECK-LABEL: @test5
void test5() {
  struct Data *const data = my_malloc(sizeof(*data));
  // CHECK: store i32 48
  gi = __builtin_object_size(data, 0);
  // CHECK: store i32 48
  gi = __builtin_object_size(data, 1);
  // CHECK: store i32 48
  gi = __builtin_object_size(data, 2);
  // CHECK: store i32 48
  gi = __builtin_object_size(data, 3);

  // CHECK: store i32 40
  gi = __builtin_object_size(&data->t[1], 0);
  // CHECK: store i32 36
  gi = __builtin_object_size(&data->t[1], 1);
  // CHECK: store i32 40
  gi = __builtin_object_size(&data->t[1], 2);
  // CHECK: store i32 36
  gi = __builtin_object_size(&data->t[1], 3);

  struct Data *const arr = my_calloc(sizeof(*data), 2);
  // CHECK: store i32 96
  gi = __builtin_object_size(arr, 0);
  // CHECK: store i32 96
  gi = __builtin_object_size(arr, 1);
  // CHECK: store i32 96
  gi = __builtin_object_size(arr, 2);
  // CHECK: store i32 96
  gi = __builtin_object_size(arr, 3);

  // CHECK: store i32 88
  gi = __builtin_object_size(&arr->t[1], 0);
  // CHECK: store i32 36
  gi = __builtin_object_size(&arr->t[1], 1);
  // CHECK: store i32 88
  gi = __builtin_object_size(&arr->t[1], 2);
  // CHECK: store i32 36
  gi = __builtin_object_size(&arr->t[1], 3);
}

// CHECK-LABEL: @test6
void test6() {
  // Things that would normally trigger conservative estimates don't need to do
  // so when we know the source of the allocation.
  struct Data *const data = my_malloc(sizeof(*data) + 10);
  // CHECK: store i32 11
  gi = __builtin_object_size(data->end, 0);
  // CHECK: store i32 11
  gi = __builtin_object_size(data->end, 1);
  // CHECK: store i32 11
  gi = __builtin_object_size(data->end, 2);
  // CHECK: store i32 11
  gi = __builtin_object_size(data->end, 3);

  struct Data *const arr = my_calloc(sizeof(*arr) + 5, 3);
  // AFAICT, GCC treats malloc and calloc identically. So, we should do the
  // same.
  //
  // Additionally, GCC ignores the initial array index when determining whether
  // we're writing off the end of an alloc_size base. e.g.
  //   arr[0].end
  //   arr[1].end
  //   arr[2].end
  // ...Are all considered "writing off the end", because there's no way to tell
  // with high accuracy if the user meant "allocate a single N-byte `Data`",
  // or "allocate M smaller `Data`s with extra padding".

  // CHECK: store i32 112
  gi = __builtin_object_size(arr->end, 0);
  // CHECK: store i32 112
  gi = __builtin_object_size(arr->end, 1);
  // CHECK: store i32 112
  gi = __builtin_object_size(arr->end, 2);
  // CHECK: store i32 112
  gi = __builtin_object_size(arr->end, 3);

  // CHECK: store i32 112
  gi = __builtin_object_size(arr[0].end, 0);
  // CHECK: store i32 112
  gi = __builtin_object_size(arr[0].end, 1);
  // CHECK: store i32 112
  gi = __builtin_object_size(arr[0].end, 2);
  // CHECK: store i32 112
  gi = __builtin_object_size(arr[0].end, 3);

  // CHECK: store i32 64
  gi = __builtin_object_size(arr[1].end, 0);
  // CHECK: store i32 64
  gi = __builtin_object_size(arr[1].end, 1);
  // CHECK: store i32 64
  gi = __builtin_object_size(arr[1].end, 2);
  // CHECK: store i32 64
  gi = __builtin_object_size(arr[1].end, 3);

  // CHECK: store i32 16
  gi = __builtin_object_size(arr[2].end, 0);
  // CHECK: store i32 16
  gi = __builtin_object_size(arr[2].end, 1);
  // CHECK: store i32 16
  gi = __builtin_object_size(arr[2].end, 2);
  // CHECK: store i32 16
  gi = __builtin_object_size(arr[2].end, 3);
}

// CHECK-LABEL: @test7
void test7() {
  struct Data *const data = my_malloc(sizeof(*data) + 5);
  // CHECK: store i32 9
  gi = __builtin_object_size(data->pad, 0);
  // CHECK: store i32 3
  gi = __builtin_object_size(data->pad, 1);
  // CHECK: store i32 9
  gi = __builtin_object_size(data->pad, 2);
  // CHECK: store i32 3
  gi = __builtin_object_size(data->pad, 3);
}

// CHECK-LABEL: @test8
void test8() {
  // Non-const pointers aren't currently supported.
  void *buf = my_calloc(100, 5);
  // CHECK: @llvm.objectsize.i64.p0i8(i8* %{{.*}}, i1 false, i1 true, i1 false)
  gi = __builtin_object_size(buf, 0);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(buf, 1);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(buf, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(buf, 3);
}

// CHECK-LABEL: @test9
void test9() {
  // Check to be sure that we unwrap things correctly.
  short *const buf0 = (my_malloc(100));
  short *const buf1 = (short*)(my_malloc(100));
  short *const buf2 = ((short*)(my_malloc(100)));

  // CHECK: store i32 100
  gi = __builtin_object_size(buf0, 0);
  // CHECK: store i32 100
  gi = __builtin_object_size(buf1, 0);
  // CHECK: store i32 100
  gi = __builtin_object_size(buf2, 0);
}

// CHECK-LABEL: @test10
void test10() {
  // Yay overflow
  short *const arr = my_calloc((size_t)-1 / 2 + 1, 2);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr, 0);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr, 1);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(arr, 3);

  // As an implementation detail, CharUnits can't handle numbers greater than or
  // equal to 2**63. Realistically, this shouldn't be a problem, but we should
  // be sure we don't emit crazy results for this case.
  short *const buf = my_malloc((size_t)-1);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(buf, 0);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(buf, 1);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(buf, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(buf, 3);

  short *const arr_big = my_calloc((size_t)-1 / 2 - 1, 2);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr_big, 0);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr_big, 1);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr_big, 2);
  // CHECK: store i32 0
  gi = __builtin_object_size(arr_big, 3);
}

void *my_tiny_malloc(char) __attribute__((alloc_size(1)));
void *my_tiny_calloc(char, char) __attribute__((alloc_size(1, 2)));

// CHECK-LABEL: @test11
void test11() {
  void *const vp = my_tiny_malloc(100);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 0);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 1);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 2);
  // CHECK: store i32 100
  gi = __builtin_object_size(vp, 3);

  // N.B. This causes char overflow, but not size_t overflow, so it should be
  // supported.
  void *const arr = my_tiny_calloc(100, 5);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 0);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 1);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 2);
  // CHECK: store i32 500
  gi = __builtin_object_size(arr, 3);
}

void *my_signed_malloc(long) __attribute__((alloc_size(1)));
void *my_signed_calloc(long, long) __attribute__((alloc_size(1, 2)));

// CHECK-LABEL: @test12
void test12() {
  // CHECK: store i32 100
  gi = __builtin_object_size(my_signed_malloc(100), 0);
  // CHECK: store i32 500
  gi = __builtin_object_size(my_signed_calloc(100, 5), 0);

  void *const vp = my_signed_malloc(-2);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(vp, 0);
  // N.B. These get lowered to -1 because the function calls may have
  // side-effects, and we can't determine the objectsize.
  // CHECK: store i32 -1
  gi = __builtin_object_size(my_signed_malloc(-2), 0);

  void *const arr1 = my_signed_calloc(-2, 1);
  void *const arr2 = my_signed_calloc(1, -2);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr1, 0);
  // CHECK: @llvm.objectsize
  gi = __builtin_object_size(arr2, 0);
  // CHECK: store i32 -1
  gi = __builtin_object_size(my_signed_calloc(1, -2), 0);
  // CHECK: store i32 -1
  gi = __builtin_object_size(my_signed_calloc(-2, 1), 0);
}
