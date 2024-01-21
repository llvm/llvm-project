// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple x86_64--windows -Oz -emit-llvm -target-feature +cx16 %s -o - \
// RUN:         | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_cc1 -ffreestanding -fms-extensions -fms-compatibility -fms-compatibility-version=17.00 \
// RUN:         -triple aarch64--windows -Oz -emit-llvm %s -o - \
// RUN:         | FileCheck %s --check-prefixes=CHECK,CHECK-AARCH64

// Ensure that we emit _Interlocked atomic operations specifying natural
// alignment, even when clang's usual alignment derivation would result in a
// lower alignment value.

// intrin.h needs size_t, but -ffreestanding prevents us from getting it from
// stddef.h.  Work around it with this typedef.
typedef __SIZE_TYPE__ size_t;

#include <intrin.h>

#pragma pack(1)
typedef struct {
  char a;
  short b;
  long c;
  long long d;
  void *p;
} X;

_Static_assert(sizeof(X) == 23, "");
_Static_assert(__alignof__(X) == 1, "");

// CHECK-LABEL: @test_InterlockedExchangePointer(
// CHECK:   atomicrmw {{.*}} align 8
void *test_InterlockedExchangePointer(X *x) {
  return _InterlockedExchangePointer(&x->p, 0);
}

// CHECK-LABEL: @test_InterlockedExchange8(
// CHECK:   atomicrmw {{.*}} align 1
char test_InterlockedExchange8(X *x) {
  return _InterlockedExchange8(&x->a, 0);
}

// CHECK-LABEL: @test_InterlockedExchange16(
// CHECK:   atomicrmw {{.*}} align 2
short test_InterlockedExchange16(X *x) {
  return _InterlockedExchange16(&x->b, 0);
}

// CHECK-LABEL: @test_InterlockedExchange(
// CHECK:   atomicrmw {{.*}} align 4
long test_InterlockedExchange(X *x) {
  return _InterlockedExchange(&x->c, 0);
}

// CHECK-LABEL: @test_InterlockedExchange64(
// CHECK:   atomicrmw {{.*}} align 8
long long test_InterlockedExchange64(X *x) {
  return _InterlockedExchange64(&x->d, 0);
}

// CHECK-LABEL: @test_InterlockedIncrement(
// CHECK:   atomicrmw {{.*}} align 4
long test_InterlockedIncrement(X *x) {
  return _InterlockedIncrement(&x->c);
}

// CHECK-LABEL: @test_InterlockedDecrement16(
// CHECK:   atomicrmw {{.*}} align 2
short test_InterlockedDecrement16(X *x) {
  return _InterlockedDecrement16(&x->b);
}


// CHECK-LABEL: @test_InterlockedCompareExchangePointer(
// CHECK:   cmpxchg {{.*}} align 8
void *test_InterlockedCompareExchangePointer(X *x) {
  return _InterlockedCompareExchangePointer(&x->p, 0, 0);
}

// CHECK-LABEL: @test_InterlockedCompareExchange8(
// CHECK:   cmpxchg {{.*}} align 1
char test_InterlockedCompareExchange8(X *x) {
  return _InterlockedCompareExchange8(&x->a, 0, 0);
}

// CHECK-LABEL: @test_InterlockedCompareExchange16(
// CHECK:   cmpxchg {{.*}} align 2
short test_InterlockedCompareExchange16(X *x) {
  return _InterlockedCompareExchange16(&x->b, 0, 0);
}

// CHECK-LABEL: @test_InterlockedCompareExchange(
// CHECK:   cmpxchg {{.*}} align 4
long test_InterlockedCompareExchange(X *x) {
  return _InterlockedCompareExchange(&x->c, 0, 0);
}

// CHECK-LABEL: @test_InterlockedCompareExchange64(
// CHECK:   cmpxchg {{.*}} align 8
long long test_InterlockedCompareExchange64(X *x) {
  return _InterlockedCompareExchange64(&x->d, 0, 0);
}

#ifdef __aarch64__
// CHECK-AARCH64-LABEL: @test_InterlockedAdd(
// CHECK-AARCH64:   atomicrmw {{.*}} align 4
long test_InterlockedAdd(X *x) {
  return _InterlockedAdd(&x->c, 4);
}
#endif
