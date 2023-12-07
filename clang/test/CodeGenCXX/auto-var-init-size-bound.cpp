// Pattern related size bound tests: 1, 8, 4096, 4097
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-size-bound=1 %s -emit-llvm -o - | FileCheck -check-prefix=PATTERN-COMMON -check-prefix=PATTERN-BOUND-1 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-size-bound=8 %s -emit-llvm -o - | FileCheck -check-prefix=PATTERN-COMMON -check-prefix=PATTERN-BOUND-8 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-size-bound=4096 %s -emit-llvm -o - | FileCheck -check-prefix=PATTERN-COMMON -check-prefix=PATTERN-BOUND-4096 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-size-bound=4097 %s -emit-llvm -o - | FileCheck -check-prefix=PATTERN-COMMON -check-prefix=PATTERN-BOUND-4097 %s
//
// Zero related size bound tests: 1, 8, 4096, 4097
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-size-bound=1 %s -emit-llvm -o - | FileCheck -check-prefix=ZERO-COMMON -check-prefix=ZERO-BOUND-1 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-size-bound=8 %s -emit-llvm -o - | FileCheck -check-prefix=ZERO-COMMON -check-prefix=ZERO-BOUND-8 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-size-bound=4096 %s -emit-llvm -o - | FileCheck -check-prefix=ZERO-COMMON -check-prefix=ZERO-BOUND-4096 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-size-bound=4097 %s -emit-llvm -o - | FileCheck -check-prefix=ZERO-COMMON -check-prefix=ZERO-BOUND-4097 %s

#define ARRLEN 1024

int foo(unsigned n) {
  bool var_size_1;
  long var_size_8 = 123;
  void *var_size_8p;
  int var_size_4096[ARRLEN];
  // VLA, non-constant size
  int var_vla[n];
  // builtin, non-constant size
  var_size_8p = __builtin_alloca(sizeof(unsigned long long) * n);
  // There are 6 variables: var_size_1, var_size_8, var_size_8p, var_size_4096,
  // var_vla, and a builtin anonymous var ("%5").
  // "var_vla" and "%5" have a non-constant size, and they should be auto-inited
  //  disregarding the size bound.
  // - Size bound 1: "var_vla" and "%5"
  // - Size bound 8: "var_size_1", "var_vla", and "%5"
  // - Size bound 4096: "var_size_1", "var_size_8", "var_size_8p",
  //                    "var_vla", and "%5"
  // - Size bound 4097: "var_size_1", "var_size_8", "var_size_8p",
  //                    "var_size_4096", "var_vla", and "%5"
  //
  // PATTERN-BOUND-1-NOT: store i8 -86, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-1-NOT: store i64 -6148914691236517206, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-1-NOT: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-1-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 -86, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // PATTERN-BOUND-8: store i8 -86, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-8-NOT: store i64 -6148914691236517206, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-8-NOT: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-8-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 -86, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // PATTERN-BOUND-4096: store i8 -86, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-4096: store i64 -6148914691236517206, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-4096: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-4096-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 -86, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // PATTERN-BOUND-4097: store i8 -86, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-4097: store i64 -6148914691236517206, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-4097: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-BOUND-4097: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 -86, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // PATTERN-COMMON: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %vla.cur, ptr align 4 @__const._Z3fooj.var_vla, i64 4, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-COMMON: call void @llvm.memset.p0.i64(ptr align 16 %5, i8 -86, i64 %mul, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-BOUND-1-NOT: store i8 0, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-1-NOT: store i64 0, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-1-NOT: store ptr null, ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-1-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 0, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-BOUND-8: store i8 0, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-8-NOT: store i64 0, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-8-NOT: store ptr null, ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-8-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 0, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-BOUND-4096: store i8 0, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-4096: store i64 0, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-4096: store ptr null, ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-4096-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 0, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-BOUND-4097: store i8 0, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-4097: store i64 0, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-4097: store ptr null, ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-BOUND-4097: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 0, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-COMMON: call void @llvm.memset.p0.i64(ptr align 16 %vla, i8 0, i64 %3, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-COMMON: call void @llvm.memset.p0.i64(ptr align 16 %5, i8 0, i64 %mul, i1 false), !annotation [[AUTO_INIT:!.+]]

  return 0;
}
