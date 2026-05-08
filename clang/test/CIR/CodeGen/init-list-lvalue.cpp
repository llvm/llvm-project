// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

struct Struct {
  int i;
  const char *c;
};

void test1(int i) {
  // CIR: cir.func {{.*}}@_Z5test1i(%[[I_ARG:.*]]: {{.*}})
  // LLVM: define {{.*}}void @_Z5test1i(i32 {{.*}}%[[I_ARG:.*]])
  int &refI = {i};
  // CIR: %[[I_ALLOCA:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
  // CIR: %[[REFI_ALLOCA:.*]] = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["refI", init, const]
  // CIR: cir.store %[[I_ARG]], %[[I_ALLOCA]] : !s32i, !cir.ptr<!s32i>
  // CIR: cir.store {{.*}}%[[I_ALLOCA]], %[[REFI_ALLOCA]] : !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>
  // LLVM: %[[I_ALLOCA:.*]] = alloca i32
  // LLVM: %[[REFI_ALLOCA:.*]] = alloca ptr
  // LLVM: store i32 %[[I_ARG]], ptr %[[I_ALLOCA]]
  // LLVM: store ptr %[[I_ALLOCA]], ptr %[[REFI_ALLOCA]]
}

void test2() {
  // CIR-LABEL: cir.func {{.*}}@_Z5test2v()
  // LLVM-LABEL: define {{.*}}void @_Z5test2v()
  Struct s {1, "asdf"};
  Struct &refS = {s};
  // CIR: %[[S_ALLOCA:.*]] = cir.alloca !rec_Struct, !cir.ptr<!rec_Struct>, ["s", init]
  // CIR: %[[REFS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>, ["refS", init, const]
  // CIR: %[[GET_S_INIT:.*]] = cir.get_global @__const._Z5test2v.s : !cir.ptr<!rec_Struct>
  // CIR: cir.copy %[[GET_S_INIT]] to %[[S_ALLOCA]] : !cir.ptr<!rec_Struct> loc(#loc33)
  // CIR: cir.store {{.*}}%[[S_ALLOCA]], %[[REFS_ALLOCA]] : !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>
  // LLVM: %[[S_ALLOCA:.*]] = alloca %struct.Struct
  // LLVM: %[[REFS_ALLOCA:.*]] = alloca ptr
  // LLVM: call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}%[[S_ALLOCA]], ptr {{.*}}@__const._Z5test2v.s, i64 16, i1 false)
  // LLVM: store ptr %[[S_ALLOCA]], ptr %[[REFS_ALLOCA]]
}

// Note: In addition to testing init-list-lvalue, this also tests
// FunctionalCastExpr.
void test3(Struct &s) {
  // CIR: cir.func {{.*}}@_Z5test3R6Struct(%[[S_ARG:.*]]: !cir.ptr<!rec_Struct> {{.*}})
  // LLVM: define dso_local void @_Z5test3R6Struct(ptr{{.*}}%[[S_ARG:.*]])
  using refSTy = Struct &;
  Struct &refS = refSTy{s};
  // CIR: %[[S_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>, ["s", init, const]
  // CIR: %[[REFS_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>, ["refS", init, const]
  // CIR: cir.store %[[S_ARG]], %[[S_ALLOCA]] : !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>
  // CIR: %[[S_LOAD:.*]] = cir.load %[[S_ALLOCA]] : !cir.ptr<!cir.ptr<!rec_Struct>>, !cir.ptr<!rec_Struct>
  // CIR: cir.store {{.*}}%[[S_LOAD]], %[[REFS_ALLOCA]] : !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>
  // LLVM: %[[S_ALLOCA:.*]] = alloca ptr
  // LLVM: %[[REFS_ALLOCA:.*]] = alloca ptr
  // LLVM: store ptr %[[S_ARG]], ptr %[[S_ALLOCA]]
  // LLVM: %[[S_LOAD:.*]] = load ptr, ptr %[[S_ALLOCA]]
  // LLVM: store ptr %[[S_LOAD]], ptr %[[REFS_ALLOCA]]
}

void test4() {
  // CIR-LABEL: cir.func {{.*}}@_Z5test4v()
  // LLVM-LABEL: define {{.*}}void @_Z5test4v()
  Struct s;
  auto& [sb1, sb2] {s};

  // CIR: %[[S_ALLOCA:.*]] = cir.alloca !rec_Struct, !cir.ptr<!rec_Struct>, ["s"] {alignment = 8 : i64}
  // CIR: %[[SB_ALLOCA:.*]] = cir.alloca !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>, ["", init, const]
  // CIR: cir.store {{.*}}%[[S_ALLOCA]], %[[SB_ALLOCA]] : !cir.ptr<!rec_Struct>, !cir.ptr<!cir.ptr<!rec_Struct>>
  // LLVM: %[[S_ALLOCA:.*]] = alloca %struct.Struct
  // LLVM: %[[SB_ALLOCA:.*]] = alloca ptr
  // LLVM: store ptr %[[S_ALLOCA]], ptr %[[SB_ALLOCA]]
}
