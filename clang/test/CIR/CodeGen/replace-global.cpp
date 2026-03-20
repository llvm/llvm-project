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

S* ptrToS = &gS;

struct R {
  R() { use(&gS); }
};
static R gR;

void use_as_constant() {
  constexpr S *ptrToS = &gS;
}

// This is just here to force ptrToS to be emitted.
S *get_ptr_to_s() {
  return ptrToS;
}

// Multi-index case: ptrToElement = &gSMulti.arr[5] creates a GlobalViewAttr with
// indices (struct member 0 = arr, array index 5). With extern gSMulti declared
// before the use, gSMulti is created when evaluating ptrToElement's initializer,
// then replaced when gSMulti is defined, so createNewGlobalView runs with multiple
// indices. Definition has external linkage to match the extern declaration.
extern S gSMulti;
char *ptrToElement = &gSMulti.arr[5];
S gSMulti = {{0x50, 0x4B, 0x03, 0x04}};

char *get_ptr_to_element() { return ptrToElement; }

// CIR: cir.global {{.*}} @_ZL2gS = #cir.const_record<{#cir.const_record<{#cir.int<80> : !s8i, #cir.int<75> : !s8i, #cir.int<3> : !s8i, #cir.int<4> : !s8i, #cir.zero : !cir.array<!s8i x 24>}> : !rec_anon_struct}> : !rec_anon_struct1
// CIR: cir.global {{.*}} @ptrToS = #cir.global_view<@_ZL2gS> : !cir.ptr<!rec_S>

// CIR: cir.func {{.*}} @_ZN1RC2Ev
// CIR:   %[[GS_PTR:.*]] = cir.get_global @_ZL2gS : !cir.ptr<!rec_anon_struct1>
// CIR:   %[[GS_AS_S:.*]] = cir.cast bitcast %[[GS_PTR]] : !cir.ptr<!rec_anon_struct1> -> !cir.ptr<!rec_S>
// CIR:   %[[GS_AS_VOID:.*]] = cir.cast bitcast %[[GS_AS_S]] : !cir.ptr<!rec_S> -> !cir.ptr<!void>
// CIR:   cir.call @_Z3usePv(%[[GS_AS_VOID]]) : (!cir.ptr<!void> {{.*}}) -> ()

// Multi-index case: ptrToElement = &gSMulti.arr[5] produces a global_view with
// multiple indices, exercising createNewGlobalView.
// CIR: cir.global {{.*}} @gSMulti = #cir.const_record<
// CIR: cir.global {{.*}} @ptrToElement = #cir.global_view<@gSMulti, [0, 4, 1]> : !cir.ptr<

// CIR: cir.func {{.*}} @_Z15use_as_constantv()
// CIR:   %[[PTR_TO_S:.*]] = cir.alloca !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>, ["ptrToS", init, const]
// CIR:   %[[GLOBAL_PTR:.*]] = cir.const #cir.global_view<@_ZL2gS> : !cir.ptr<!rec_S>
// CIR:   cir.store{{.*}} %[[GLOBAL_PTR]], %[[PTR_TO_S]] : !cir.ptr<!rec_S>, !cir.ptr<!cir.ptr<!rec_S>>

// LLVM: @_ZL2gS = internal global { <{ i8, i8, i8, i8, [24 x i8] }> } { <{ i8, i8, i8, i8, [24 x i8] }> <{ i8 80, i8 75, i8 3, i8 4, [24 x i8] zeroinitializer }> }, align 1
// LLVM: @ptrToS = global ptr @_ZL2gS, align 8
// LLVM: @gSMulti = global {{.*}} align 1
// LLVM: @ptrToElement = global ptr getelementptr

// LLVM: define {{.*}} void @_ZN1RC2Ev
// LLVM:   call void @_Z3usePv(ptr noundef @_ZL2gS)

// LLVM: define {{.*}} void @_Z15use_as_constantv()
// LLVM:   %[[PTR_TO_S:.*]] = alloca ptr
// LLVM:   store ptr @_ZL2gS, ptr %[[PTR_TO_S]]

// OGCG: @ptrToS = global ptr @_ZL2gS, align 8
// OGCG: @ptrToElement = global ptr {{.*}} align 8
// OGCG: @gSMulti = global {{.*}} align 1
// OGCG: @_ZL2gS = internal global { <{ i8, i8, i8, i8, [24 x i8] }> } { <{ i8, i8, i8, i8, [24 x i8] }> <{ i8 80, i8 75, i8 3, i8 4, [24 x i8] zeroinitializer }> }, align 1

// OGCG: define {{.*}} void @_Z15use_as_constantv()
// OGCG:   %[[PTR_TO_S:.*]] = alloca ptr
// OGCG:   store ptr @_ZL2gS, ptr %[[PTR_TO_S]]

// OGCG: define {{.*}} void @_ZN1RC2Ev
// OGCG:   call void @_Z3usePv(ptr noundef @_ZL2gS)
