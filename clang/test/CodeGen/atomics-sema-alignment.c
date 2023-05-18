// RUN: %clang_cc1 -triple aarch64-linux-gnu %s -emit-llvm -o /dev/null -verify

typedef struct {
  int a, b;
} IntPair;

typedef struct {
  long long a;
} LongStruct;

typedef int __attribute__((aligned(1))) unaligned_int;

void func(IntPair *p) {
  IntPair res;
  __atomic_load(p, &res, 0);                    // expected-warning {{misaligned atomic operation may incur significant performance penalty; the expected alignment (8 bytes) exceeds the actual alignment (4 bytes)}}
  __atomic_store(p, &res, 0);                   // expected-warning {{misaligned atomic operation may incur significant performance penalty; the expected alignment (8 bytes) exceeds the actual alignment (4 bytes)}}
  __atomic_fetch_add((unaligned_int *)p, 1, 2); // expected-warning {{misaligned atomic operation may incur significant performance penalty; the expected alignment (4 bytes) exceeds the actual alignment (1 bytes)}}
  __atomic_fetch_sub((unaligned_int *)p, 1, 3); // expected-warning {{misaligned atomic operation may incur significant performance penalty; the expected alignment (4 bytes) exceeds the actual alignment (1 bytes)}}
}

void func1(LongStruct *p) {
  LongStruct res;
  __atomic_load(p, &res, 0);
  __atomic_store(p, &res, 0);
  __atomic_fetch_add((int *)p, 1, 2);
  __atomic_fetch_sub((int *)p, 1, 3);
}

typedef struct {
  void *a;
  void *b;
} Foo;

typedef struct {
  void *a;
  void *b;
  void *c;
  void *d;
} __attribute__((aligned(32))) ThirtyTwo;

void braz(Foo *foo, ThirtyTwo *braz) {
  Foo bar;
  __atomic_load(foo, &bar, __ATOMIC_RELAXED); // expected-warning {{misaligned atomic operation may incur significant performance penalty; the expected alignment (16 bytes) exceeds the actual alignment (8 bytes)}}

  ThirtyTwo thirtyTwo1;
  ThirtyTwo thirtyTwo2;
  __atomic_load(&thirtyTwo1, &thirtyTwo2, __ATOMIC_RELAXED); // expected-warning {{large atomic operation may incur significant performance penalty; the access size (32 bytes) exceeds the max lock-free size (16  bytes)}}
}
