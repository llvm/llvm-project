// FIXME: pthread_create() definition in Builtins.td doesn't match the real one, so it doesn't get recognized as a builtin and attributes aren't added.
// RUN: false
// XFAIL: *

// RUN: %clang_cc1 %s -S -emit-llvm -o - -disable-llvm-optzns | FileCheck %s

// CHECK: declare !callback ![[cid:[0-9]+]] {{.*}}i32 @pthread_create
// CHECK: ![[cid]] = !{![[cidb:[0-9]+]]}
// CHECK: ![[cidb]] = !{i64 2, i64 3, i1 false}

// Taken from test/Analysis/retain-release.m
//{
struct _opaque_pthread_t {};
struct _opaque_pthread_attr_t {};
typedef struct _opaque_pthread_t *__darwin_pthread_t;
typedef struct _opaque_pthread_attr_t __darwin_pthread_attr_t;
typedef __darwin_pthread_t pthread_t;
typedef __darwin_pthread_attr_t pthread_attr_t;

int pthread_create(pthread_t *, const pthread_attr_t *,
                   void *(*)(void *), void *);
//}

const int GlobalVar = 0;

static void *callee0(void *payload) {
  return payload;
}

static void *callee1(void *payload) {
  return payload;
}

void foo() {
  pthread_t MyFirstThread;
  pthread_create(&MyFirstThread, 0, callee0, 0);

  pthread_t MySecondThread;
  pthread_create(&MySecondThread, 0, callee1, (void *)&GlobalVar);
}
