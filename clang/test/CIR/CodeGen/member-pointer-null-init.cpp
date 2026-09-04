// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM,LLVMCIR --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM,OGCG --input-file=%t.ll %s

struct Inner {
  int Inner::*p;
};

struct Outer {
  Inner a;
  int b;
};

// Value-init of a heap-allocated struct containing a pointer-to-data-member.
// The member pointer is null (-1), so the stored constant must carry -1.

// CIR-LABEL: cir.func {{.*}}@_Z8make_newv
// CIR:         [[NULL:%.*]] = cir.const #cir.const_record<{#cir.int<-1> : !s64i}> : !rec_Inner
// CIR:         cir.store align(8) [[NULL]], {{%.*}} : !rec_Inner, !cir.ptr<!rec_Inner>

// LLVMCIR-LABEL: define {{.*}} ptr @_Z8make_newv
// LLVMCIR:         call {{.*}} @_Znwm
// LLVMCIR:         store %struct.Inner { i64 -1 }, ptr %{{.*}}, align 8

// OGCG: @{{.*}} = private constant %struct.Inner { i64 -1 }
// OGCG-LABEL: define {{.*}} ptr @_Z8make_newv
// OGCG:         call {{.*}} @llvm.memcpy{{.*}}i64 8

Inner *make_new() { return new Inner(); }

// Partial aggregate init: Inner subobject 'a' is value-initialized because
// it has no designated initializer.

// CIR-LABEL: cir.func {{.*}}@_Z11runtime_aggi
// CIR:         cir.const #cir.int<-1> : !s64i
// CIR:         cir.store align(8) {{%.*}}, {{%.*}} : !s64i

// LLVM-LABEL: define {{.*}} void @_Z11runtime_aggi
// LLVM:          store i64 -1, ptr %{{.*}}, align 8

void runtime_agg(int x) {
  Outer o = {.b = x};
  (void)o;
}
