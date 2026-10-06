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

// Arrays of pointer-to-data-member should be all -1s.

// CIR: cir.global "private" internal dso_local @_ZZ12static_slotsvE8fn_slots = #cir.const_array<[#cir.int<-1> : !s64i, #cir.int<-1> : !s64i]> : !cir.array<!s64i x 2>
// CIR: cir.global external @ns_slots = #cir.const_array<[#cir.int<-1> : !s64i, #cir.int<-1> : !s64i]> : !cir.array<!s64i x 2>

// LLVM-DAG: @_ZZ12static_slotsvE8fn_slots = internal global [2 x i64] [i64 -1, i64 -1]
// LLVM-DAG: @ns_slots = {{.*}}global [2 x i64] [i64 -1, i64 -1]

int Inner::*ns_slots[2];

void static_slots() {
  static int Inner::*fn_slots[2];
  (void)fn_slots;
}

// CIR: cir.global external @vol_slots = #cir.const_array<[#cir.int<-1> : !s64i, #cir.int<-1> : !s64i]> : !cir.array<!s64i x 2>

// LLVM-DAG: @vol_slots = {{.*}}global [2 x i64] [i64 -1, i64 -1]

volatile int Inner::*vol_slots[2];

// Array of record-types with the member pointer, also should have -1s.

// CIR: cir.global external @rec_slots = #cir.const_array<[#cir.const_record<{#cir.int<-1> : !s64i}> : !rec_Inner, #cir.const_record<{#cir.int<-1> : !s64i}> : !rec_Inner]> : !cir.array<!rec_Inner x 2>

// LLVM-DAG: @rec_slots = {{.*}}global [2 x %struct.Inner] [%struct.Inner { i64 -1 }, %struct.Inner { i64 -1 }]

Inner rec_slots[2];

// A nested (multi-dimensional) array of pointers-to-data-member must have
// -1 in every innermost element.

// CIR: cir.global external @md_slots = #cir.const_array<[#cir.const_array<[#cir.int<-1> : !s64i, #cir.int<-1> : !s64i, #cir.int<-1> : !s64i]> : !cir.array<!s64i x 3>, #cir.const_array<[#cir.int<-1> : !s64i, #cir.int<-1> : !s64i, #cir.int<-1> : !s64i]> : !cir.array<!s64i x 3>]> : !cir.array<!cir.array<!s64i x 3> x 2>

// LLVM-DAG: @md_slots = {{.*}}global [2 x [3 x i64]] [{{\[3 x i64\]}} [i64 -1, i64 -1, i64 -1], {{\[3 x i64\]}} [i64 -1, i64 -1, i64 -1]]

int Inner::*md_slots[2][3];

// Same with 'new' allocated types.

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

// Aggregate init should also get this right.

// CIR-LABEL: cir.func {{.*}}@_Z11runtime_aggi
// CIR:         cir.const #cir.int<-1> : !s64i
// CIR:         cir.store align(8) {{%.*}}, {{%.*}} : !s64i

// LLVM-LABEL: define {{.*}} void @_Z11runtime_aggi
// LLVM:          store i64 -1, ptr %{{.*}}, align 8

void runtime_agg(int x) {
  Outer o = {.b = x};
  (void)o;
}
