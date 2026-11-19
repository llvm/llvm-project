// RUN: %clang_cc1 -std=c17 -Wno-deprecated-non-prototype -triple spir64 -disable-llvm-passes -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c17 -Wno-deprecated-non-prototype -triple spir64 -disable-llvm-passes -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c17 -Wno-deprecated-non-prototype -triple spir64 -disable-llvm-passes -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

// Plain C calls on SPIR use the non-default spir_func calling convention for
// direct, indirect and unprototyped calls.

int callee(int x);

int direct(int x) { return callee(x); }

// CIR: cir.func {{.*}}@direct({{.*}}) -> !s32i cc(spir_function)
// CIR:   cir.call @callee(%{{.*}}) cc(spir_function) : (!s32i {{.*}}) -> !s32i

// LLVM: define {{.*}}spir_func i32 @direct(
// LLVM:   call spir_func i32 @callee(i32 noundef %{{.*}})

int indirect(int (*fp)(int), int x) { return fp(x); }

// CIR: cir.func {{.*}}@indirect({{.*}}) -> !s32i cc(spir_function)
// CIR:   cir.call %{{.*}}(%{{.*}}) cc(spir_function) : (!cir.ptr<!cir.func<(!s32i) -> !s32i>>, !s32i {{.*}}) -> !s32i

// LLVM: define {{.*}}spir_func i32 @indirect(
// LLVM:   call spir_func i32 %{{.*}}(i32 noundef %{{.*}})

int noproto();

int call_noproto(void) { return noproto(42); }

int noproto(int x) { return x; }

// CIR: cir.func {{.*}}@call_noproto() -> !s32i cc(spir_function)
// CIR:   %[[FN:.*]] = cir.get_global @noproto
// CIR:   cir.call %[[FN]](%{{.*}}) cc(spir_function)
// CIR: cir.func {{.*}}no_proto {{.*}}@noproto({{.*}}) -> !s32i cc(spir_function)

// LLVM: define {{.*}}spir_func i32 @call_noproto(
// LLVM:   call spir_func i32 @noproto(i32 noundef 42)
