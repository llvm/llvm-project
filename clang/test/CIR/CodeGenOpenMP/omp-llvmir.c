// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// CIR-LABEL: cir.func no_inline no_proto dso_local @main() -> !s32i {
// CIR: [[RETVAL:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"] {alignment = 4 : i64}
// CIR: [[J:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["j"] {alignment = 4 : i64}
// CIR:   omp.parallel {
// CIR:     cir.scope {
// CIR:       [[I:%.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR:       [[ZERO1:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:       cir.store align(4) [[ZERO1]], [[I]] : !s32i, !cir.ptr<!s32i>
// CIR:       cir.for : cond {
// CIR:         [[LOAD1:%.*]] = cir.load align(4) [[I]] : !cir.ptr<!s32i>, !s32i
// CIR:         [[LIMIT:%.*]] = cir.const #cir.int<10000> : !s32i
// CIR:         [[CMP:%.*]] = cir.cmp(lt, [[LOAD1]], [[LIMIT]]) : !s32i, !cir.bool
// CIR:         cir.condition([[CMP]])
// CIR:       } body {
// CIR:         [[ZERO2:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:         cir.store align(4) [[ZERO2]], [[J]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       } step {
// CIR:         [[LOAD2:%.*]] = cir.load align(4) [[I]] : !cir.ptr<!s32i>, !s32i
// CIR:         [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:         [[ADD:%.*]] = cir.binop(add, [[LOAD2]], [[ONE]]) nsw : !s32i
// CIR:         cir.store align(4) [[ADD]], [[I]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       }
// CIR:     }
// CIR:     omp.terminator
// CIR:   }
// CIR:   [[ZERO3:%.*]] = cir.const #cir.int<0> : !s32i
// CIR:   cir.store [[ZERO3]], [[RETVAL]] : !s32i, !cir.ptr<!s32i>
// CIR:   [[RET:%.*]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.return [[RET]] : !s32i
// CIR: }

// LLVM-LABEL: define dso_local i32 @main()
// LLVM: %[[STRUCTARG:.*]] = alloca { ptr, ptr }, align 8
// LLVM: %[[VAR1:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[VAR2:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[VAR3:.*]] = alloca i32, i64 1, align 4
// LLVM: br label %[[ENTRY:.*]]

// LLVM: [[ENTRY]]:
// LLVM: br label %[[OMP_PARALLEL:.*]]

// LLVM: [[OMP_PARALLEL]]:
// LLVM: %[[GEP1:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
// LLVM: store ptr %[[VAR1]], ptr %[[GEP1]], align 8
// LLVM: %[[GEP2:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
// LLVM: store ptr %[[VAR3]], ptr %[[GEP2]], align 8
// LLVM: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @main..omp_par, ptr %[[STRUCTARG]])
// LLVM: br label %[[OMP_PAR_EXIT:.*]]

// LLVM: [[OMP_PAR_EXIT]]:
// LLVM: store i32 0, ptr %[[VAR2]], align 4
// LLVM: %[[LOAD:.*]] = load i32, ptr %[[VAR2]], align 4
// LLVM: ret i32 %[[LOAD]]

// LLVM-LABEL: define internal void @main..omp_par(ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr %{{.*}})
// LLVM: [[PAR_ENTRY:.*]]:
// LLVM: %[[GEP_A:.*]] = getelementptr { ptr, ptr }, ptr %{{.*}}, i32 0, i32 0
// LLVM: %[[LOAD_A:.*]] = load ptr, ptr %[[GEP_A]], align 8
// LLVM: %[[GEP_B:.*]] = getelementptr { ptr, ptr }, ptr %{{.*}}, i32 0, i32 1
// LLVM: %[[LOAD_B:.*]] = load ptr, ptr %[[GEP_B]], align 8
// LLVM: %[[TID_LOCAL:.*]] = alloca i32, align 4
// LLVM: %[[TID_VAL:.*]] = load i32, ptr %{{.*}}, align 4
// LLVM: store i32 %[[TID_VAL]], ptr %[[TID_LOCAL]], align 4
// LLVM: %{{.*}} = load i32, ptr %[[TID_LOCAL]], align 4
// LLVM: br label %[[AFTER_ALLOCA:.*]]

// LLVM: [[AFTER_ALLOCA]]:
// LLVM: br label %[[PAR_REGION:.*]]

// LLVM: [[PAR_REGION]]:
// LLVM: br label %[[PAR_REGION1:.*]]

// LLVM: [[PAR_REGION1]]:
// LLVM: br label %[[PAR_REGION2:.*]]

// LLVM: [[PAR_REGION2]]:
// LLVM: store i32 0, ptr %[[LOAD_A]], align 4
// LLVM: br label %[[PAR_REGION3:.*]]

// LLVM: [[PAR_REGION3]]:
// LLVM: %[[I_LOAD:.*]] = load i32, ptr %[[LOAD_A]], align 4
// LLVM: %[[CMP:.*]] = icmp slt i32 %[[I_LOAD]], 10000
// LLVM: br i1 %[[CMP]], label %[[PAR_REGION4:.*]], label %[[PAR_REGION6:.*]]

// LLVM: [[PAR_REGION6]]:
// LLVM: br label %[[PAR_REGION7:.*]]

// LLVM: [[PAR_REGION7]]:
// LLVM: br label %[[REGION_CONT:.*]]

// LLVM: [[REGION_CONT]]:
// LLVM: br label %[[PRE_FINALIZE:.*]]

// LLVM: [[PRE_FINALIZE]]:
// LLVM: br label %[[FINI:.*]]

// LLVM: [[FINI]]:
// LLVM: br label %[[EXIT_STUB:.*]]

// LLVM: [[PAR_REGION4]]:
// LLVM: store i32 0, ptr %[[LOAD_B]], align 4
// LLVM: br label %[[PAR_REGION5:.*]]

// LLVM: [[PAR_REGION5]]:
// LLVM: %[[I_LOAD2:.*]] = load i32, ptr %[[LOAD_A]], align 4
// LLVM: %[[ADD:.*]] = add nsw i32 %[[I_LOAD2]], 1
// LLVM: store i32 %[[ADD]], ptr %[[LOAD_A]], align 4
// LLVM: br label %[[PAR_REGION3]]

// LLVM: [[EXIT_STUB]]:
// LLVM: ret void

// OGCG-LABEL: define dso_local i32 @main()
// OGCG: [[ENTRY:.*]]:
// OGCG: %[[RETVAL:.*]] = alloca i32, align 4
// OGCG: %[[J:.*]] = alloca i32, align 4
// OGCG: store i32 0, ptr %[[RETVAL]], align 4
// OGCG: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @main.omp_outlined, ptr %[[J]])
// OGCG: ret i32 0

// OGCG-LABEL: define internal void @main.omp_outlined(ptr noalias noundef %{{.*}}, ptr noalias noundef %{{.*}}, ptr noundef nonnull align 4 dereferenceable(4) %{{.*}})
// OGCG: [[OUTLINED_ENTRY:.*]]:
// OGCG: %[[GLOBAL_TID_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[BOUND_TID_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[J_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[I:.*]] = alloca i32, align 4
// OGCG: store ptr %{{.*}}, ptr %[[GLOBAL_TID_ADDR]], align 8
// OGCG: store ptr %{{.*}}, ptr %[[BOUND_TID_ADDR]], align 8
// OGCG: store ptr %{{.*}}, ptr %[[J_ADDR]], align 8
// OGCG: %[[J_LOAD:.*]] = load ptr, ptr %[[J_ADDR]], align 8
// OGCG: store i32 0, ptr %[[I]], align 4
// OGCG: br label %[[FOR_COND:.*]]

// OGCG: [[FOR_COND]]:
// OGCG: %[[I_LOAD:.*]] = load i32, ptr %[[I]], align 4
// OGCG: %[[CMP:.*]] = icmp slt i32 %[[I_LOAD]], 10000
// OGCG: br i1 %[[CMP]], label %[[FOR_BODY:.*]], label %[[FOR_END:.*]]

// OGCG: [[FOR_BODY]]:
// OGCG: store i32 0, ptr %[[J_LOAD]], align 4
// OGCG: br label %[[FOR_INC:.*]]

// OGCG: [[FOR_INC]]:
// OGCG: %[[I_LOAD2:.*]] = load i32, ptr %[[I]], align 4
// OGCG: %[[ADD:.*]] = add nsw i32 %[[I_LOAD2]], 1
// OGCG: store i32 %[[ADD]], ptr %[[I]], align 4
// OGCG: br label %[[FOR_COND]]

// OGCG: [[FOR_END]]:
// OGCG: ret void

int main() {
  int j;
#pragma omp parallel
  for (int i = 0; i < 10000; i=i+1)
    j = 0;

  return 0;
}
