// RUN: %clang_cc1 -triple i386-unknown-unknown -fopenmp %s -emit-llvm -o - -disable-llvm-optzns | FileCheck %s --check-prefix=RUN1

// RUN1-DAG: @broker0({{[^#]*#[0-9]+}} !callback ![[cid0:[0-9]+]]
__attribute__((callback(1, 2))) void *broker0(void *(*callee)(void *), void *payload) {
  return callee(payload);
}

// RUN1-DAG: @broker1({{[^#]*#[0-9]+}} !callback ![[cid1:[0-9]+]]
__attribute__((callback(callee, payload))) void *broker1(void *payload, void *(*callee)(void *)) {
  return broker0(callee, payload);
}

void *broker2(void (*callee)(void));

// RUN1-DAG: declare !callback ![[cid2:[0-9]+]] i8* @broker2
__attribute__((callback(callee))) void *broker2(void (*callee)(void));

void *broker2(void (*callee)(void));

// RUN1-DAG: declare !callback ![[cid3:[0-9]+]] i8* @broker3
__attribute__((callback(4, 1, 2, c))) void *broker3(int, int, int c, int (*callee)(int, int, int), int);

// RUN1-DAG: declare !callback ![[cid4:[0-9]+]] i8* @broker4
__attribute__((callback(4, -1, a, __))) void *broker4(int a, int, int, int (*callee)(int, int, int), int);

// RUN1-DAG: declare !callback ![[cid5:[0-9]+]] i8* @broker5
__attribute__((callback(4, d, 5, 2))) void *broker5(int, int, int, int (*callee)(int, int, int), int d);

static void *VoidPtr2VoidPtr(void *payload) {
  return payload;
}

static int ThreeInt2Int(int a, int b, int c) {
  return a * b + c;
}

void foo(void) {
  broker0(VoidPtr2VoidPtr, 0l);
  broker1(0l, VoidPtr2VoidPtr);
  broker2(foo);
  broker3(1, 4, 5, ThreeInt2Int, 1);
  broker4(4, 2, 7, ThreeInt2Int, 0);
  broker5(8, 0, 3, ThreeInt2Int, 4);
}

// RUN1-DAG: ![[cid0]] = !{![[cid0b:[0-9]+]]}
// RUN1-DAG: ![[cid0b]] = !{i64 0, i64 1, i1 false}
// RUN1-DAG: ![[cid1]] = !{![[cid1b:[0-9]+]]}
// RUN1-DAG: ![[cid1b]] = !{i64 1, i64 0, i1 false}
// RUN1-DAG: ![[cid2]] = !{![[cid2b:[0-9]+]]}
// RUN1-DAG: ![[cid2b]] = !{i64 0, i1 false}
// RUN1-DAG: ![[cid3]] = !{![[cid3b:[0-9]+]]}
// RUN1-DAG: ![[cid3b]] = !{i64 3, i64 0, i64 1, i64 2, i1 false}
// RUN1-DAG: ![[cid4]] = !{![[cid4b:[0-9]+]]}
// RUN1-DAG: ![[cid4b]] = !{i64 3, i64 -1, i64 0, i64 -1, i1 false}
// RUN1-DAG: ![[cid5]] = !{![[cid5b:[0-9]+]]}
// RUN1-DAG: ![[cid5b]] = !{i64 3, i64 4, i64 4, i64 1, i1 false}
