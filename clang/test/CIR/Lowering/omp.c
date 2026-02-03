// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fopenmp -emit-cir %s -o %t.cir
// RUN: cir-opt %t.cir -cir-to-llvm -o - | FileCheck %s -check-prefix=MLIR
// RUN: cir-translate %t.cir -cir-to-llvmir --target x86_64-unknown-linux-gnu --disable-cc-lowering  | FileCheck %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fopenmp -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck %s -check-prefix=CIR --input-file %t-cir.ll
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -emit-llvm %s -o %t.ll
// RUN: FileCheck %s -check-prefix=OGCG --input-file %t.ll

// MLIR-LABEL: llvm.func @main() -> i32
// MLIR-SAME: attributes {dso_local, no_inline, no_proto}
// MLIR: %[[C1:.*]] = llvm.mlir.constant(1 : i64) : i64
// MLIR: %[[ALLOCA1:.*]] = llvm.alloca %[[C1]] x i32 {alignment = 4 : i64}
// MLIR: %[[ALLOCA2:.*]] = llvm.alloca %{{.*}} x i32 {alignment = 4 : i64}
// MLIR: %[[ALLOCA3:.*]] = llvm.alloca %{{.*}} x i32 {alignment = 4 : i64}
// MLIR: omp.parallel {
// MLIR: llvm.br ^bb{{[0-9]+}}
// MLIR: ^bb{{[0-9]+}}:
// MLIR: %[[ZERO1:.*]] = llvm.mlir.constant(0 : i32) : i32
// MLIR: llvm.store %[[ZERO1]], %{{.*}} {alignment = 4 : i64}
// MLIR: llvm.br ^bb{{[0-9]+}}
// MLIR: ^bb{{[0-9]+}}:
// MLIR: %[[LOAD1:.*]] = llvm.load %{{.*}} {alignment = 4 : i64}
// MLIR: %[[C10000:.*]] = llvm.mlir.constant(10000 : i32) : i32
// MLIR: %[[CMP:.*]] = llvm.icmp "slt" %[[LOAD1]], %[[C10000]] : i32
// MLIR: llvm.cond_br %[[CMP]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// MLIR: ^bb{{[0-9]+}}:
// MLIR: %[[ZERO2:.*]] = llvm.mlir.constant(0 : i32) : i32
// MLIR: llvm.store %[[ZERO2]], %{{.*}} {alignment = 4 : i64}
// MLIR: llvm.br ^bb{{[0-9]+}}
// MLIR: ^bb{{[0-9]+}}:
// MLIR: %[[LOAD2:.*]] = llvm.load %{{.*}} {alignment = 4 : i64}
// MLIR: %[[C1_I32:.*]] = llvm.mlir.constant(1 : i32) : i32
// MLIR: %[[ADD:.*]] = llvm.add %[[LOAD2]], %[[C1_I32]] overflow<nsw> : i32
// MLIR: llvm.store %[[ADD]], %{{.*}} {alignment = 4 : i64}
// MLIR: llvm.br ^bb{{[0-9]+}}
// MLIR: ^bb{{[0-9]+}}:
// MLIR: llvm.br ^bb{{[0-9]+}}
// MLIR: ^bb{{[0-9]+}}:
// MLIR: omp.terminator
// MLIR: %[[ZERO3:.*]] = llvm.mlir.constant(0 : i32) : i32
// MLIR: llvm.store %[[ZERO3]], %{{.*}} {alignment = 4 : i64}
// MLIR: %[[RETVAL:.*]] = llvm.load %{{.*}} {alignment = 4 : i64}
// MLIR: llvm.return %[[RETVAL]] : i32

// CIR-LABEL: define dso_local i32 @main()
// CIR: %[[STRUCTARG:.*]] = alloca { ptr, ptr }, align 8
// CIR: %[[VAR1:.*]] = alloca i32, i64 1, align 4
// CIR: %[[VAR2:.*]] = alloca i32, i64 1, align 4
// CIR: %[[VAR3:.*]] = alloca i32, i64 1, align 4
// CIR: br label %[[ENTRY:.*]]

// CIR: [[ENTRY]]:
// CIR: %[[THREAD_NUM:.*]] = call i32 @__kmpc_global_thread_num(ptr @1)
// CIR: br label %[[OMP_PARALLEL:.*]]

// CIR: [[OMP_PARALLEL]]:
// CIR: %[[GEP1:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 0
// CIR: store ptr %[[VAR1]], ptr %[[GEP1]], align 8
// CIR: %[[GEP2:.*]] = getelementptr { ptr, ptr }, ptr %[[STRUCTARG]], i32 0, i32 1
// CIR: store ptr %[[VAR3]], ptr %[[GEP2]], align 8
// CIR: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @1, i32 1, ptr @main..omp_par, ptr %[[STRUCTARG]])
// CIR: br label %[[OMP_PAR_EXIT:.*]]

// CIR: [[OMP_PAR_EXIT]]:
// CIR: store i32 0, ptr %[[VAR2]], align 4
// CIR: %[[LOAD:.*]] = load i32, ptr %[[VAR2]], align 4
// CIR: ret i32 %[[LOAD]]

// CIR-LABEL: define internal void @main..omp_par(ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr %{{.*}})
// CIR: [[PAR_ENTRY:.*]]:
// CIR: %[[GEP_A:.*]] = getelementptr { ptr, ptr }, ptr %{{.*}}, i32 0, i32 0
// CIR: %[[LOAD_A:.*]] = load ptr, ptr %[[GEP_A]], align 8
// CIR: %[[GEP_B:.*]] = getelementptr { ptr, ptr }, ptr %{{.*}}, i32 0, i32 1
// CIR: %[[LOAD_B:.*]] = load ptr, ptr %[[GEP_B]], align 8
// CIR: %[[TID_LOCAL:.*]] = alloca i32, align 4
// CIR: %[[TID_VAL:.*]] = load i32, ptr %{{.*}}, align 4
// CIR: store i32 %[[TID_VAL]], ptr %[[TID_LOCAL]], align 4
// CIR: %{{.*}} = load i32, ptr %[[TID_LOCAL]], align 4
// CIR: br label %[[AFTER_ALLOCA:.*]]

// CIR: [[AFTER_ALLOCA]]:
// CIR: br label %[[PAR_REGION:.*]]

// CIR: [[PAR_REGION]]:
// CIR: br label %[[PAR_REGION1:.*]]

// CIR: [[PAR_REGION1]]:
// CIR: br label %[[PAR_REGION2:.*]]

// CIR: [[PAR_REGION2]]:
// CIR: store i32 0, ptr %[[LOAD_A]], align 4
// CIR: br label %[[PAR_REGION3:.*]]

// CIR: [[PAR_REGION3]]:
// CIR: %[[I_LOAD:.*]] = load i32, ptr %[[LOAD_A]], align 4
// CIR: %[[CMP:.*]] = icmp slt i32 %[[I_LOAD]], 10000
// CIR: br i1 %[[CMP]], label %[[PAR_REGION4:.*]], label %[[PAR_REGION6:.*]]

// CIR: [[PAR_REGION6]]:
// CIR: br label %[[PAR_REGION7:.*]]

// CIR: [[PAR_REGION7]]:
// CIR: br label %[[REGION_CONT:.*]]

// CIR: [[REGION_CONT]]:
// CIR: br label %[[PRE_FINALIZE:.*]]

// CIR: [[PRE_FINALIZE]]:
// CIR: br label %[[FINI:.*]]

// CIR: [[FINI]]:
// CIR: br label %[[EXIT_STUB:.*]]

// CIR: [[PAR_REGION4]]:
// CIR: store i32 0, ptr %[[LOAD_B]], align 4
// CIR: br label %[[PAR_REGION5:.*]]

// CIR: [[PAR_REGION5]]:
// CIR: %[[I_LOAD2:.*]] = load i32, ptr %[[LOAD_A]], align 4
// CIR: %[[ADD:.*]] = add nsw i32 %[[I_LOAD2]], 1
// CIR: store i32 %[[ADD]], ptr %[[LOAD_A]], align 4
// CIR: br label %[[PAR_REGION3]]

// CIR: [[EXIT_STUB]]:
// CIR: ret void

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
