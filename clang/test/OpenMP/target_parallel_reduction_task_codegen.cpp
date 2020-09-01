// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-unknown-linux -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple x86_64-unknown-linux -fexceptions -fcxx-exceptions -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=50 -x c++ -emit-llvm %s -fexceptions -fcxx-exceptions -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -x c++ -triple x86_64-unknown-unknown -fexceptions -fcxx-exceptions -debug-info-kind=limited -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: @main
int main(int argc, char **argv) {
#pragma omp target parallel reduction(task, +: argc, argv[0:10][0:argc])
  {
#pragma omp task in_reduction(+: argc, argv[0:10][0:argc])
    ;
  }
}

// CHECK: call void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_call(%struct.ident_t* @{{.+}}, i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i32*, i8**)* [[OUTLINED:@.+]] to void (i32*, i32*, ...)*), i32* %{{.+}}, i8** %{{.+}})

// CHECK: define internal void [[OUTLINED]](i32* noalias %{{.+}}, i32* noalias %{{.+}}, i32* {{.+}}, i8** {{.+}})
// CHECK: [[ARGC_FP_ADDR:%.+]] = alloca i32,
// CHECK: [[TR:%.+]] = alloca [2 x %struct.kmp_taskred_input_t],
// CHECK: [[TG:%.+]] = alloca i8*,

// Init firstprivate copy of argc
// CHECK: store i32 0, i32* [[ARGC_FP_ADDR]],
// CHECK: [[ARGV_FP_ADDR:%.+]] = alloca i8, i64 [[SIZE:%.+]],
// CHECK: store i64 [[SIZE]], i64* [[SIZE_ADDR:%.+]],

// Init firstprivate copy of argv[0:10][0:argc]
// CHECK: [[END:%.+]] = getelementptr i8, i8* [[ARGV_FP_ADDR]], i64 [[SIZE]]
// CHECK: [[EMPTY:%.+]] = icmp eq i8* [[ARGV_FP_ADDR]], [[END]]
// CHECK: br i1 [[EMPTY]], label %[[DONE:.+]], label %[[INIT:.+]]
// CHECK: [[INIT]]:
// CHECK: [[EL:%.+]] = phi i8* [ [[ARGV_FP_ADDR]], %{{.+}} ], [ [[NEXT_EL:%.+]], %[[INIT]] ]
// CHECK: store i8 0, i8* [[EL]],
// CHECK: [[NEXT_EL:%.+]] = getelementptr i8, i8* [[EL]], i32 1
// CHECK: [[FINISHED:%.+]] = icmp eq i8* [[NEXT_EL]], [[END]]
// CHECK: br i1 [[FINISHED]], label %[[DONE]], label %[[INIT]]
// CHECK: [[DONE]]:

// Register task reduction.
// CHECK: [[TR0_ADDR:%.+]] = getelementptr inbounds [2 x %struct.kmp_taskred_input_t], [2 x %struct.kmp_taskred_input_t]* [[TR]], i64 0, i64 0
// CHECK: [[TR0_SHARED_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR0_ADDR]], i32 0, i32 0
// CHECK: [[BC:%.+]] = bitcast i32* [[ARGC_FP_ADDR]] to i8*
// CHECK: store i8* [[BC]], i8** [[TR0_SHARED_ADDR]],
// CHECK: [[TR0_ORIG_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR0_ADDR]], i32 0, i32 1
// CHECK: [[BC:%.+]] = bitcast i32* %{{.+}} to i8*
// CHECK: store i8* [[BC]], i8** [[TR0_ORIG_ADDR]],
// CHECK: [[TR0_SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR0_ADDR]], i32 0, i32 2
// CHECK: store i64 4, i64* [[TR0_SIZE_ADDR]],
// CHECK: [[TR0_INIT_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR0_ADDR]], i32 0, i32 3
// CHECK: store i8* bitcast (void (i8*, i8*)* [[ARGC_INIT:@.+]] to i8*), i8** [[TR0_INIT_ADDR]],
// CHECK: [[TR0_FINI_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR0_ADDR]], i32 0, i32 4
// CHECK: store i8* null, i8** [[TR0_FINI_ADDR]],
// CHECK: [[TR0_COMB_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR0_ADDR]], i32 0, i32 5
// CHECK: store i8* bitcast (void (i8*, i8*)* [[ARGC_COMB:@.+]] to i8*), i8** [[TR0_COMB_ADDR]],
// CHECK: [[TR0_FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR0_ADDR]], i32 0, i32 6
// CHECK: [[BC:%.+]] = bitcast i32* [[TR0_FLAGS_ADDR]] to i8*
// CHECK: call void @llvm.memset.p0i8.i64(i8* {{.*}}[[BC]], i8 0, i64 4, i1 false)
// CHECK: [[TR1_ADDR:%.+]] = getelementptr inbounds [2 x %struct.kmp_taskred_input_t], [2 x %struct.kmp_taskred_input_t]* [[TR]], i64 0, i64 1
// CHECK: [[TR1_SHARED_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR1_ADDR]], i32 0, i32 0
// CHECK: store i8* [[ARGV_FP_ADDR]], i8** [[TR1_SHARED_ADDR]],
// CHECK: [[TR1_ORIG_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR1_ADDR]], i32 0, i32 1
// CHECK: store i8* %{{.+}}, i8** [[TR1_ORIG_ADDR]],
// CHECK: [[TR1_SIZE_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR1_ADDR]], i32 0, i32 2
// CHECK: store i64 %{{.+}}, i64* [[TR1_SIZE_ADDR]],
// CHECK: [[TR1_INIT_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR1_ADDR]], i32 0, i32 3
// CHECK: store i8* bitcast (void (i8*, i8*)* [[ARGV_INIT:@.+]] to i8*), i8** [[TR1_INIT_ADDR]],
// CHECK: [[TR1_FINI_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR1_ADDR]], i32 0, i32 4
// CHECK: store i8* null, i8** [[TR1_FINI_ADDR]],
// CHECK: [[TR1_COMB_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR1_ADDR]], i32 0, i32 5
// CHECK: store i8* bitcast (void (i8*, i8*)* [[ARGV_COMB:@.+]] to i8*), i8** [[TR1_COMB_ADDR]],
// CHECK: [[TR1_FLAGS_ADDR:%.+]] = getelementptr inbounds %struct.kmp_taskred_input_t, %struct.kmp_taskred_input_t* [[TR1_ADDR]], i32 0, i32 6
// CHECK: store i32 1, i32* [[TR1_FLAGS_ADDR]],
// CHECK: [[BC:%.+]] = bitcast [2 x %struct.kmp_taskred_input_t]* [[TR]] to i8*
// CHECK: [[TG_VAL:%.+]] = call i8* @__kmpc_taskred_modifier_init(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 0, i32 2, i8* [[BC]])
// CHECK: store i8* [[TG_VAL]], i8** [[TG]],

// CHECK: [[PTR:%.+]] = call i8* @__kmpc_omp_task_alloc(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 1, i64 48, i64 24, i32 (i32, i8*)* bitcast (i32 (i32, [[TASK_TY:%.+]]*)* [[TASK:@.+]] to i32 (i32, i8*)*))
// CHECK: [[TASK_DATA_ADDR:%.+]] = bitcast i8* [[PTR]] to [[TASK_TY]]*
// CHECK: [[PRIVATES_ADDR:%.+]] = getelementptr inbounds [[TASK_TY]], [[TASK_TY]]* [[TASK_DATA_ADDR]], i32 0, i32 1
// CHECK: [[TG_PRIV_ADDR:%.+]] = getelementptr inbounds [[TASK_PRIVATES_TY:%.+]], %{{.+}}* [[PRIVATES_ADDR]], i32 0, i32 0
// CHECK: [[TG_VAL:%.+]] = load i8*, i8** [[TG]],
// CHECK: store i8* [[TG_VAL]], i8** [[TG_PRIV_ADDR]],

// CHECK: call i32 @__kmpc_omp_task(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i8* [[PTR]])

// CHECK: call void @__kmpc_task_reduction_modifier_fini(%struct.ident_t* @{{.+}}, i32 %{{.+}}, i32 0)
// CHECK: call i32 @__kmpc_reduce_nowait(

// CHECK: define internal void [[ARGC_INIT]](i8* noalias %{{.+}}, i8* noalias %{{.+}})
// CHECK: store i32 0, i32* %{{.+}},

// CHECK: define internal void [[ARGC_COMB]](i8* %{{.+}}, i8* %{{.+}})
// CHECK: [[ADD:%.+]] = add nsw i32 %{{.+}}, %{{.+}}
// CHECK: store i32 [[ADD]], i32* %{{.+}},

// CHECK: define internal void [[ARGV_INIT]](i8* noalias %{{.+}}, i8* noalias %{{.+}})
// CHECK: phi i8*
// CHECK: store i8 0, i8* [[EL:%.+]],
// CHECK: getelementptr i8, i8* [[EL]], i32 1

// CHECK: define internal void [[ARGV_COMB]](i8* %{{.+}}, i8* %{{.+}})
// CHECK: phi i8*
// CHECK: [[ADD:%.+]] = add nsw i32 %{{.+}}, %{{.+}}
// CHECK: [[CONV:%.+]] = trunc i32 [[ADD]] to i8
// CHECK: store i8 [[CONV]], i8* [[EL:%.+]],
// CHECK: getelementptr i8, i8* [[EL]], i32 1

// CHECK: define internal {{.*}}i32 [[TASK]](i32 {{.+}}, [[TASK_TY]]* {{.+}})
// CHECK-DAG: call i8* @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, i8* [[TG:%.+]], i8* [[ARGC_REF:%.+]])
// CHECK_DAG: [[TG]] = load i8*, i8** [[TG_ADDR:%.+]],
// CHECK-DAG: [[ARGC_REF]] = bitcast i32* [[ARGC_ADDR:%.+]] to i8*
// CHECK-DAG: [[ARGC_ADDR]] = load i32*, i32** [[ARGC_ADDR_REF:%.+]],
// CHECK-DAG: [[ARGC_ADDR_REF]] = getelementptr inbounds [[CAPS_TY:%.+]], %{{.+}}* [[CAP:%.+]], i32 0, i32 1
// CHECK-DAG: call i8* @__kmpc_task_reduction_get_th_data(i32 %{{.+}}, i8* [[TG:%.+]], i8* [[ARGV_REF:%.+]])
// CHECK_DAG: [[TG]] = load i8*, i8** [[TG_ADDR]],
// CHECK-DAG: [[ARGV_REF]] = load i8*, i8** [[ARGV_ADDR:%.+]],
// CHECK-DAG: [[ARGV_ADDR]] = load i8**, i8*** [[ARGV_ADDR_REF:%.+]],
// CHECK-DAG: [[ARGV_ADDR_REF]] = getelementptr inbounds [[CAPS_TY]], [[CAPS_TY]]* [[CAP]], i32 0, i32 2

#endif
