// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

union Union { int i; float f; };

float getF();

void toUnion() {
  // CIR-LABEL: toUnion()
  // LLVM-LABEL: toUnion()
  //
  (void)(union Union) getF();
  // CIR: %[[F_FUNC:.*]] = cir.get_global @getF : !cir.ptr<!cir.func<(...) -> !cir.float>>
  // CIR:  %[[F_CAST:.*]] = cir.cast bitcast %[[F_FUNC]] : !cir.ptr<!cir.func<(...) -> !cir.float>> -> !cir.ptr<!cir.func<() -> !cir.float>>
  // CIR: cir.call %[[F_CAST:.*]]() : (!cir.ptr<!cir.func<() -> !cir.float>>) -> !cir.float
  // CIR: cir.return

  // LLVM: call float @getF()
  // LLVM: ret void

  // OGCG-LABEL: define dso_local void @toUnion()
  // OGCG: call float (...) @getF()
  // OGCG: ret void
}

// GCC assign-from-cast extension:
union Union toUnionAssign() {
  // CIR-LABEL: cir.func{{.*}} @toUnionAssign() -> !s32i
  // LLVM-LABEL: define dso_local i32 @toUnionAssign()
  // OGCG-LABEL: define dso_local i32 @toUnionAssign()
  //
  // CIR: %[[COERCE:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!rec_Union>
  // CIR: %[[RET_ALLOCA:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<!rec_Union>
  // LLVM: %[[COERCE:.*]] = alloca %union.Union
  // LLVM: %[[RET_ALLOCA:.*]] = alloca %union.Union
  // OGCG: %[[RET_ALLOCA:.*]] = alloca %union.Union
  union Union u;
  u = (union Union)42;
  // CIR: %[[UNION_TO_INT:.*]] = cir.cast bitcast %[[RET_ALLOCA]] : !cir.ptr<!rec_Union> -> !cir.ptr<!s32i>
  // CIR: %[[INT42:.*]] = cir.const #cir.int<42>
  // CIR: cir.store align(4) %[[INT42]], %[[UNION_TO_INT]] : !s32i, !cir.ptr<!s32i>
  // LLVM: store i32 42, ptr %[[RET_ALLOCA]]
  // OGCG: store i32 42, ptr %[[RET_ALLOCA]]
  return u;
  // CIR: %[[LOAD:.*]] = cir.load %[[RET_ALLOCA]] : !cir.ptr<!rec_Union>, !rec_Union
  // CIR: cir.store %[[LOAD]], %[[COERCE]] : !rec_Union, !cir.ptr<!rec_Union>
  // CIR: %[[COERCE_TO_INT:.*]] = cir.cast bitcast %[[COERCE]] : !cir.ptr<!rec_Union> -> !cir.ptr<!s32i>
  // CIR: %[[RET:.*]] = cir.load %[[COERCE_TO_INT]] : !cir.ptr<!s32i>, !s32i
  // CIR: cir.return %[[RET]] : !s32i
  // LLVM: %[[LOAD:.*]] = load %union.Union, ptr %[[RET_ALLOCA]]
  // LLVM: store %union.Union %[[LOAD]], ptr %[[COERCE]]
  // LLVM: %[[RET:.*]] = load i32, ptr %[[COERCE]]
  // LLVM: ret i32 %[[RET]]
  // OGCG: %[[DIVE:.*]] = getelementptr inbounds nuw %union.Union, ptr %[[RET_ALLOCA]], i32 0, i32 0
  // OGCG: %[[RET:.*]] = load i32, ptr %[[DIVE]]
  // OGCG: ret i32 %[[RET]]
}
