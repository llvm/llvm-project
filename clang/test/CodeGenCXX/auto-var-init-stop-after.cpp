// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-stop-after=1 %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN-STOP-AFTER-1-SCALAR
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-stop-after=2 %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN-STOP-AFTER-2-ARRAY
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-stop-after=3 %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN-STOP-AFTER-3-VLA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-stop-after=4 %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN-STOP-AFTER-4-POINTER
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-stop-after=5 %s -emit-llvm -o - | FileCheck %s -check-prefix=PATTERN-STOP-AFTER-5-BUILTIN
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-stop-after=1 %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO-STOP-AFTER-1-SCALAR
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-stop-after=2 %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO-STOP-AFTER-2-ARRAY
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-stop-after=3 %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO-STOP-AFTER-3-VLA
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-stop-after=4 %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO-STOP-AFTER-4-POINTER
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-stop-after=5 %s -emit-llvm -o - | FileCheck %s -check-prefix=ZERO-STOP-AFTER-5-BUILTIN

#define ARRLEN 10

typedef struct {
  int i;
  char c;
} S;

int foo(unsigned n) {
  // scalar variable
  long a;
  // array
  S arr[ARRLEN];
  // VLA
  S vla[n];
  // pointer
  void *p;
  // builtin
  p = __builtin_alloca(sizeof(unsigned long long) * n);
  // PATTERN-STOP-AFTER-1-SCALAR:             store i64 -6148914691236517206, ptr %a, align 8
  // PATTERN-STOP-AFTER-1-SCALAR-NOT:         call void @llvm.memset.p0.i64(ptr align 16 %0, i8 -86, i64 80, i1 false)
  // PATTERN-STOP-AFTER-2-ARRAY:         call void @llvm.memset.p0.i64(ptr align 16 %arr, i8 -86, i64 80, i1 false)
  // PATTERN-STOP-AFTER-2-ARRAY-NOT:          vla-init.loop:
  // PATTERN-STOP-AFTER-3-VLA:                vla-init.loop:
  // PATTERN-STOP-AFTER-3-VLA-NEXT:           %vla.cur = phi ptr [ %vla, %vla-setup.loop ], [ %vla.next, %vla-init.loop ]
  // PATTERN-STOP-AFTER-3-VLA-NEXT-NEXT:      call void @llvm.memcpy.p0.p0.i64(ptr align 8 %vla.cur, ptr align 4 @__const._Z3fooj.vla, i64 8, i1 false)
  // PATTERN-STOP-AFTER-3-VLA-NOT:            store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %p, align 8
  // PATTERN-STOP-AFTER-4-POINTER:            store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %p, align 8
  // PATTERN-STOP-AFTER-4-POINTER-NOT:        call void @llvm.memset.p0.i64(ptr align 16 %5, i8 -86, i64 %mul, i1 false)
  // PATTERN-STOP-AFTER-5-BUILTIN:            call void @llvm.memset.p0.i64(ptr align 16 %5, i8 -86, i64 %mul, i1 false)
  // PATTERN-STOP-AFTER-5-BUILTIN-MESSAGES:   warning: -ftrivial-auto-var-init-stop-after=5 has been enabled to limit the number of times ftrivial-auto-var-init=pattern gets applied.

  // ZERO-STOP-AFTER-1-SCALAR:                store i64 0, ptr %a, align 8
  // ZERO-STOP-AFTER-1-SCALAR-NOT:            call void @llvm.memset.p0.i64(ptr align 16 %arr, i8 0, i64 80, i1 false)
  // ZERO-STOP-AFTER-2-ARRAY:            call void @llvm.memset.p0.i64(ptr align 16 %arr, i8 0, i64 80, i1 false)
  // ZERO-STOP-AFTER-2-ARRAY-NOT:             %call void @llvm.memset.p0.i64(ptr align 16 %3, i8 0, i64 %2, i1 false)
  // ZERO-STOP-AFTER-3-VLA:              call void @llvm.memset.p0.i64(ptr align 16 %vla, i8 0, i64 %3, i1 false)
  // ZERO-STOP-AFTER-3-VLA-NOT:               store ptr null, ptr %p, align 8
  // ZERO-STOP-AFTER-4-POINTER:               store ptr null, ptr %p, align 8
  // ZERO-STOP-AFTER-4-POINTER-NOT:           call void @llvm.memset.p0.i64(ptr align 16 %4, i8 0, i64 %mul, i1 false)
  // ZERO-STOP-AFTER-5-BUILTIN:               %5 = alloca i8, i64 %mul, align 16
  // ZERO-STOP-AFTER-5-BUILTIN-NEXT:          call void @llvm.memset.p0.i64(ptr align 16 %5, i8 0, i64 %mul, i1 false)
  // ZERO-STOP-AFTER-5-BUILTIN-MESSAGES:      warnings: -ftrivial-auto-var-init-stop-after=5 has been enabled to limit the number of times ftrivial-auto-var-init=zero gets applied.
  return 0;
}
