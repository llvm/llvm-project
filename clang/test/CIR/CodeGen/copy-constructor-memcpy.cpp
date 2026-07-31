// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm %s -o %t.og.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.og.ll %s

// Test that trivial copy constructors are inlined as aggregate copies
// (memcpy-equivalent special members) rather than emitted as function calls.

struct S {
  int a;
  int b;
};

struct W {
  S s;
  W(const S &src) : s(src) {}
};

void test(const S &src) {
  W w(src);
}

// The copy of S in W's constructor should be inlined as cir.copy,
// not a call to S's copy constructor.

// CIR-LABEL: cir.func{{.*}} @_ZN1WC2ERK1S
// CIR-NOT:     cir.call @_ZN1SC
// CIR:         cir.copy %{{.+}} to %{{.+}} : !cir.ptr<!rec_S>

// Both CIR-lowered LLVM and OG produce memcpy for the inlined copy.

// LLVM-LABEL: define{{.*}} void @_ZN1WC2ERK1S
// LLVM:         call void @llvm.memcpy.p0.p0.i64({{.*}}i64 8

// OGCG-LABEL: define{{.*}} void @_ZN1WC2ERK1S
// OGCG:         call void @llvm.memcpy.p0.p0.i64({{.*}}i64 8
