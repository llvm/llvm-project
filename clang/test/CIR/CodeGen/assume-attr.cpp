// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,BOTH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=OGCG,BOTH


bool hasSideEffect();

constexpr int globalVar = 5;

void assume1(), assume2(), assume3();

void usage(int i, int j) {
  // CIR: cir.func{{.*}}@_Z5usageii
  // BOTH: define{{.*}}@_Z5usageii

  // CIR: %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
  // CIR: %[[J:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["j", init]
  // CIR: %[[LOCAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["local", init, const]
  //
  // BOTH: %[[I:.*]] = alloca i32
  // BOTH: %[[J:.*]] = alloca i32
  // BOTH: %[[LOCAL:.*]] = alloca i32

  [[assume(globalVar == i)]]
    ;

  // CIR: %[[GET_GLOB_VAR:.*]] = cir.get_global @_ZL9globalVar
  // CIR: %[[LOAD_GLOB_VAR:.*]] = cir.load {{.*}}%[[GET_GLOB_VAR]]
  // CIR: %[[LOAD_I:.*]] = cir.load {{.*}}%[[I]]
  // CIR: %[[CMP:.*]] = cir.cmp(eq, %[[LOAD_GLOB_VAR]], %[[LOAD_I]]) 
  // CIR: cir.assume %[[CMP]] : !cir.bool
  // LLVM: %[[LOAD_GLOB_VAR:.*]] = load i32, ptr @_ZL9globalVar
  // BOTH: %[[LOAD_I:.*]] = load i32, ptr %[[I]]
  // LLVM: %[[CMP:.*]] = icmp eq i32 %[[LOAD_GLOB_VAR]], %[[LOAD_I]]
  // OGCG: %[[CMP:.*]] = icmp eq i32 5, %[[LOAD_I]]
  // BOTH: call void @llvm.assume(i1 %[[CMP]])
  assume1();
  // CIR: cir.call @_Z7assume1v()
  // BOTH: call void @_Z7assume1v()

  constexpr int local = 12;
  [[assume(j == i && j > local)]]
    ;
  // CIR: %[[LOAD_J:.*]] = cir.load {{.*}}%[[J]]
  // CIR: %[[LOAD_I:.*]] = cir.load {{.*}}%[[I]]
  // CIR: %[[J_EQ_I:.*]] = cir.cmp(eq, %[[LOAD_J]], %[[LOAD_I]])
  // CIR: %[[TERN:.*]] = cir.ternary(%[[J_EQ_I]], true {
  // CIR-NEXT: %[[LOAD_J:.*]] = cir.load {{.*}}%[[J]]
  // CIR-NEXT: %[[LOAD_LOCAL:.*]] = cir.load {{.*}}%[[LOCAL]]
  // CIR-NEXT: %[[J_GT_LOCAL:.*]] = cir.cmp(gt, %[[LOAD_J]], %[[LOAD_LOCAL]])
  // CIR-NEXT: cir.yield %[[J_GT_LOCAL]]
  // CIR-NEXT: }, false {
  // CIR-NEXT: %[[FALSE:.*]] = cir.const #false
  // CIR-NEXT: cir.yield %[[FALSE]]
  // CIR-NEXT: }) : (!cir.bool) -> !cir.bool
  // CIR: cir.assume %[[TERN:.*]]
  // 
  // BOTH: %[[LOAD_J:.*]] = load i32, ptr %[[J]]
  // BOTH: %[[LOAD_I:.*]] = load i32, ptr %[[I]]
  // BOTH: %[[J_EQ_I:.*]] = icmp eq i32 %[[LOAD_J]], %[[LOAD_I]]
  // BOTH: br i1 %[[J_EQ_I]], label %[[TRUE:.*]], label %[[FALSE:.*]]
  // BOTH: [[TRUE]]:
  // BOTH: %[[LOAD_J:.*]] = load i32, ptr %[[J]]
  // LLVM: %[[LOAD_LOCAL:.*]] = load i32, ptr %[[LOCAL]]
  // LLVM: %[[J_GT_LOCAL:.*]] = icmp sgt i32 %[[LOAD_J]], %[[LOAD_LOCAL]]
  // OGCG: %[[J_GT_LOCAL:.*]] = icmp sgt i32 %[[LOAD_J]], 12
  // LLVM: br label %[[DONE:.*]]
  // OGCG: br label %[[FALSE]]
  // LLVM: [[FALSE]]:
  // LLVM: br label %[[DONE]]
  // LLVM: [[DONE]]:
  // OGCG: [[FALSE]]:
  // LLVM: %[[TERN:.*]] = phi i1 [ false, %[[FALSE]] ], [ %[[J_GT_LOCAL]], %[[TRUE]] ]
  // OGCG: %[[TERN:.*]] = phi i1 [ false, %entry ], [ %[[J_GT_LOCAL]], %[[TRUE]] ]
  // BOTH: call void @llvm.assume(i1 %[[TERN]])
  assume2();
  // CIR: cir.call @_Z7assume2v()
  // BOTH: call void @_Z7assume2v()

  [[assume(hasSideEffect())]]
    ;
  // CIR-NOT: cir.assume
  // BOTH-NOT: call void @llvm.assume
  assume3();
  // CIR: cir.call @_Z7assume3v()
  // BOTH: call void @_Z7assume3v()
}

