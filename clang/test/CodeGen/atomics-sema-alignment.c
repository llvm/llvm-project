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
  __atomic_load(p, &res, 0); // expected-warning {{misaligned atomic operation may incur significant performance penalty}}
  __atomic_store(p, &res, 0); // expected-warning {{misaligned atomic operation may incur significant performance penalty}}
  __atomic_fetch_add((unaligned_int *)p, 1, 2); // expected-warning {{misaligned atomic operation may incur significant performance penalty}}
  __atomic_fetch_sub((unaligned_int *)p, 1, 3); // expected-warning {{misaligned atomic operation may incur significant performance penalty}}
}

void func1(LongStruct *p) {
  LongStruct res;
  __atomic_load(p, &res, 0);
  __atomic_store(p, &res, 0);
  __atomic_fetch_add((int *)p, 1, 2);
  __atomic_fetch_sub((int *)p, 1, 3);
}
