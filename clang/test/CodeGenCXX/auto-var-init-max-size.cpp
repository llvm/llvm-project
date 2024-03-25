// Pattern related max size tests: 1, 1024, 4096
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-max-size=1 %s -emit-llvm -o - | FileCheck -check-prefix=PATTERN-COMMON -check-prefix=PATTERN-MAX-1 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-max-size=1024 %s -emit-llvm -o - | FileCheck -check-prefix=PATTERN-COMMON -check-prefix=PATTERN-MAX-1024 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=pattern -ftrivial-auto-var-init-max-size=4096 %s -emit-llvm -o - | FileCheck -check-prefix=PATTERN-COMMON -check-prefix=PATTERN-MAX-4096 %s
//
// Zero related max size tests: 1, 1024, 4096
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-max-size=1 %s -emit-llvm -o - | FileCheck -check-prefix=ZERO-COMMON -check-prefix=ZERO-MAX-1 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-max-size=1024 %s -emit-llvm -o - | FileCheck -check-prefix=ZERO-COMMON -check-prefix=ZERO-MAX-1024 %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -ftrivial-auto-var-init=zero -ftrivial-auto-var-init-max-size=4096 %s -emit-llvm -o - | FileCheck -check-prefix=ZERO-COMMON -check-prefix=ZERO-MAX-4096 %s

struct Foo {
    int x; // we should try to make sure X is initialized.
    char buff[1024];  // this one is fine to skip
};

int foo(unsigned n) {
  bool var_size_1;
  long var_size_8 = 123;
  void *var_size_8p;
  int var_size_1024[256];
  Foo var_size_1028;
  int var_size_4096[1024];
  // VLA, non-constant size
  int var_vla[n];
  // builtin, non-constant size
  var_size_8p = __builtin_alloca(sizeof(unsigned long long) * n);
  // There are 8 variables: var_size_1, var_size_8, var_size_8p, var_size_1024,
  // var_size_1028, var_size_4096, var_vla, and a builtin anonymous var ("%5").
  // - COMMON (auto-init regardless of the max size): "var_vla", and "%5"
  // - Max size 1: "var_size_1"
  // - Max size 1024: "var_size_1", "var_size_8", "var_size_8p", "var_size_1024"
  // - Max size 4096: "var_size_1", "var_size_8", "var_size_8p", "var_size_1024", "var_size_1028", "var_size_4096"
  //
  // PATTERN-MAX-1: store i8 -86, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1-NOT: store i64 -6148914691236517206, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1-NOT: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_1024, i8 -86, i64 1024, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1-NOT: call void @llvm.memset.p0.i64(ptr align 4 %var_size_1028, i8 -86, i64 1028, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 -86, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // PATTERN-MAX-1024: store i8 -86, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1024: store i64 -6148914691236517206, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1024: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1024: call void @llvm.memset.p0.i64(ptr align 16 %var_size_1024, i8 -86, i64 1024, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1024-NOT: call void @llvm.memset.p0.i64(ptr align 4 %var_size_1028, i8 -86, i64 1028, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-1024-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 -86, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // PATTERN-MAX-4096: store i8 -86, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-4096: store i64 -6148914691236517206, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-4096: store ptr inttoptr (i64 -6148914691236517206 to ptr), ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-4096: call void @llvm.memset.p0.i64(ptr align 16 %var_size_1024, i8 -86, i64 1024, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-4096: call void @llvm.memset.p0.i64(ptr align 4 %var_size_1028, i8 -86, i64 1028, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-MAX-4096: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 -86, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // PATTERN-COMMON: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %vla.cur, ptr align 4 @__const._Z3fooj.var_vla, i64 4, i1 false), !annotation [[AUTO_INIT:!.+]]
  // PATTERN-COMMON: call void @llvm.memset.p0.i64(ptr align 16 %5, i8 -86, i64 %mul, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-MAX-1: store i8 0, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1-NOT: store i64 0, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1-NOT: store ptr null, ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_1024, i8 0, i64 1024, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1-NOT: call void @llvm.memset.p0.i64(ptr align 4 %var_size_1028, i8 0, i64 1028, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 0, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-MAX-1024: store i8 0, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1024: store i64 0, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1024: store ptr null, ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1024: call void @llvm.memset.p0.i64(ptr align 16 %var_size_1024, i8 0, i64 1024, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1024-NOT: call void @llvm.memset.p0.i64(ptr align 4 %var_size_1028, i8 0, i64 1028, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-1024-NOT: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 0, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-MAX-4096: store i8 0, ptr %var_size_1, align 1, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-4096: store i64 0, ptr %var_size_8, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-4096: store ptr null, ptr %var_size_8p, align 8, !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-4096: call void @llvm.memset.p0.i64(ptr align 16 %var_size_1024, i8 0, i64 1024, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-4096: call void @llvm.memset.p0.i64(ptr align 4 %var_size_1028, i8 0, i64 1028, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-MAX-4096: call void @llvm.memset.p0.i64(ptr align 16 %var_size_4096, i8 0, i64 4096, i1 false), !annotation [[AUTO_INIT:!.+]]

  // ZERO-COMMON: call void @llvm.memset.p0.i64(ptr align 16 %vla, i8 0, i64 %3, i1 false), !annotation [[AUTO_INIT:!.+]]
  // ZERO-COMMON: call void @llvm.memset.p0.i64(ptr align 16 %5, i8 0, i64 %mul, i1 false), !annotation [[AUTO_INIT:!.+]]

  return 0;
}
