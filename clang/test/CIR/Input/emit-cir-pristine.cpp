// -emit-cir=pristine prints CIRGen's output before any ClangIR pass runs.
// Compiling that output resumes the pipeline and gives the same result as
// compiling the source directly.

// REQUIRES: x86-registered-target

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir=pristine %s -o %t.cir
// RUN: FileCheck %s --input-file=%t.cir

// CHECK: cir.global external @gs = #cir.zero : !rec_S ctor {
// CHECK: tls_refs = <"_ZTW2tl", "_ZTH2tl"> @tl
// CHECK: cir.func {{.*}} @_Z4addvDv2_fS_(%arg0: !cir.vector<2 x !cir.float> {{.*}} requires_call_conv_lowering

// It is the same as -clangir-disable-passes, and implies -fclangir.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-passes -emit-cir %s -o %t.disable.cir
// RUN: diff %t.cir %t.disable.cir
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-cir=pristine %s -o %t.implied.cir
// RUN: diff %t.cir %t.implied.cir

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.direct.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %t.cir -o %t.ll
// RUN: diff %t.direct.ll %t.ll

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-obj %s -o %t.direct.o
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-obj %t.cir -o %t.o
// RUN: cmp %t.direct.o %t.o

// RUN: not %clang_cc1 -fclangir -emit-cir=bogus %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD
// BAD: error: invalid value 'bogus' in '-emit-cir=bogus'

// The keyword is a -cc1 option only.
// RUN: not %clang -fclangir -emit-cir=pristine -c %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s --check-prefix=DRIVER
// DRIVER: error: unknown argument '-emit-cir=pristine'

typedef float v2f __attribute__((vector_size(8)));
v2f addv(v2f a, v2f b) { return a + b; }

struct S {
  S();
  int x;
};
S gs;

thread_local int tl = 3;
int get() { return tl; }
