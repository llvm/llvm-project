// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct S {
  char arr[28];
};

void use(void*);

static S gS = {{0x50, 0x4B, 0x03, 0x04}};

struct R {
  R() { use(&gS); }
};
static R gR;

// CIR: cir.func {{.*}} @_ZN1RC2Ev
// CIR:   %[[GS_PTR:.*]] = cir.get_global @_ZL2gS : !cir.ptr<!rec_anon_struct1>
// CIR:   %[[GS_AS_S:.*]] = cir.cast bitcast %[[GS_PTR]] : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!rec_S>
// CIR:   %[[GS_AS_VOID:.*]] = cir.cast bitcast %[[GS_AS_S]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:   cir.call @_Z3usePv(%[[GS_AS_VOID]]) : (!cir.ptr<!void> {{.*}}) -> ()

// CIR: cir.global {{.*}} @_ZL2gS = #cir.const_record<{#cir.const_record<{#cir.int<80> : !s8i, #cir.int<75> : !s8i, #cir.int<3> : !s8i, #cir.int<4> : !s8i, #cir.zero : !cir.array<!s8i x 24>}> : !rec_anon_struct}> : !rec_anon_struct1

// LLVM: @_ZL2gS = internal global { <{ i8, i8, i8, i8, [24 x i8] }> } { <{ i8, i8, i8, i8, [24 x i8] }> <{ i8 80, i8 75, i8 3, i8 4, [24 x i8] zeroinitializer }> }, align 1

// LLVM: define {{.*}} void @_ZN1RC2Ev
// LLVM:   call void @_Z3usePv(ptr noundef @_ZL2gS)

// OGCG: @_ZL2gS = internal global { <{ i8, i8, i8, i8, [24 x i8] }> } { <{ i8, i8, i8, i8, [24 x i8] }> <{ i8 80, i8 75, i8 3, i8 4, [24 x i8] zeroinitializer }> }, align 1

// OGCG: define {{.*}} void @_ZN1RC2Ev
// OGCG:   call void @_Z3usePv(ptr noundef @_ZL2gS)
