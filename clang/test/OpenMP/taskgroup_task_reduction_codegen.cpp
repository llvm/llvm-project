// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - -triple x86_64-apple-darwin10 | FileCheck %s --check-prefix=CHECK --check-prefix=DEBUG

// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ %s -verify -debug-info-kind=limited -emit-llvm -o - -triple x86_64-apple-darwin10 | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics
#ifndef HEADER
#define HEADER

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_null_allocator;
extern const omp_allocator_handle_t omp_default_mem_alloc;
extern const omp_allocator_handle_t omp_large_cap_mem_alloc;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_high_bw_mem_alloc;
extern const omp_allocator_handle_t omp_low_lat_mem_alloc;
extern const omp_allocator_handle_t omp_cgroup_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;
extern const omp_allocator_handle_t omp_thread_mem_alloc;

// CHECK-DAG: @reduction_size.[[ID:.+]]_[[CID:[0-9]+]].artificial.
// CHECK-DAG: @reduction_size.[[ID]]_[[CID]].artificial..cache.

struct S {
  int a;
  S() : a(0) {}
  S(const S&) {}
  S& operator=(const S&) {return *this;}
  ~S() {}
  friend S operator+(const S&a, const S&b) {return a;}
};

int main(int argc, char **argv) {
  int a;
  float b;
  S c[5];
  short d[argc];
#pragma omp taskgroup allocate(omp_pteam_mem_alloc: a) task_reduction(+: a, b, argc)
  {
#pragma omp taskgroup task_reduction(-:c, d)
    ;
  }
  return 0;
}
// CHECK-LABEL: @main
// CHECK:       alloca i32,
// CHECK:       [[ARGC_ADDR:%.+]] = alloca i32,
// CHECK:       [[ARGV_ADDR:%.+]] = alloca i8**,
// CHECK:       [[A:%.+]] = alloca i32,
// CHECK:       [[B:%.+]] = alloca float,
// CHECK:       [[C:%.+]] = alloca [5 x %struct.S],
// CHECK:       [[RD_IN1:%.+]] = alloca [3 x [[T1:%[^,]+]]],
// CHECK:       [[TD1:%.+]] = alloca i8*,
// CHECK:       [[RD_IN2:%.+]] = alloca [2 x [[T2:%[^,]+]]],
// CHECK:       [[TD2:%.+]] = alloca i8*,

// CHECK:       [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(%struct.ident_t*
// CHECK:       [[VLA:%.+]] = alloca i16, i64 [[VLA_SIZE:%[^,]+]],

// CHECK:       call void @__kmpc_taskgroup(%struct.ident_t* {{[^,]+}}, i32 [[GTID]])
// CHECK-DAG:   [[BC_A:%.+]] = bitcast i32* [[A]] to i8*
// CHECK-DAG:   store i8* [[BC_A]], i8** [[A_REF:[^,]+]],
// CHECK-DAG:   [[A_REF]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPA:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[BC_A:%.+]] = bitcast i32* [[A]] to i8*
// CHECK-DAG:   store i8* [[BC_A]], i8** [[A_REF:[^,]+]],
// CHECK-DAG:   [[A_REF]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPA]], i32 0, i32 1
// CHECK-DAG:   [[GEPA]] = getelementptr inbounds [3 x [[T1]]], [3 x [[T1]]]* [[RD_IN1]], i64 0, i64
// CHECK-DAG:   [[TMP6:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPA]], i32 0, i32 2
// CHECK-DAG:   store i64 4, i64* [[TMP6]],
// CHECK-DAG:   [[TMP7:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPA]], i32 0, i32 3
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[AINIT:.+]] to i8*), i8** [[TMP7]],
// CHECK-DAG:   [[TMP8:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPA]], i32 0, i32 4
// CHECK-DAG:   store i8* null, i8** [[TMP8]],
// CHECK-DAG:   [[TMP9:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPA]], i32 0, i32 5
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[ACOMB:.+]] to i8*), i8** [[TMP9]],
// CHECK-DAG:   [[TMP10:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPA]], i32 0, i32 6
// CHECK-DAG:   [[TMP11:%.+]] = bitcast i32* [[TMP10]] to i8*
// CHECK-DAG:   call void @llvm.memset.p0i8.i64(i8* align 8 [[TMP11]], i8 0, i64 4, i1 false)
// CHECK-DAG:   [[TMP13:%.+]] = bitcast float* [[B]] to i8*
// CHECK-DAG:   store i8* [[TMP13]], i8** [[TMP12:%[^,]+]],
// CHECK-DAG:   [[TMP12]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPB:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[TMP13:%.+]] = bitcast float* [[B]] to i8*
// CHECK-DAG:   store i8* [[TMP13]], i8** [[TMP12:%[^,]+]],
// CHECK-DAG:   [[TMP12]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPB]], i32 0, i32 1
// CHECK-DAG:   [[GEPB]] = getelementptr inbounds [3 x [[T1]]], [3 x [[T1]]]* [[RD_IN1]], i64 0, i64
// CHECK-DAG:   [[TMP14:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPB]], i32 0, i32 2
// CHECK-DAG:   store i64 4, i64* [[TMP14]],
// CHECK-DAG:   [[TMP15:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPB]], i32 0, i32 3
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[BINIT:.+]] to i8*), i8** [[TMP15]],
// CHECK-DAG:   [[TMP16:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPB]], i32 0, i32 4
// CHECK-DAG:   store i8* null, i8** [[TMP16]],
// CHECK-DAG:   [[TMP17:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPB]], i32 0, i32 5
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[BCOMB:.+]] to i8*), i8** [[TMP17]],
// CHECK-DAG:   [[TMP18:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPB]], i32 0, i32 6
// CHECK-DAG:   [[TMP19:%.+]] = bitcast i32* [[TMP18]] to i8*
// CHECK-DAG:   call void @llvm.memset.p0i8.i64(i8* align 8 [[TMP19]], i8 0, i64 4, i1 false)
// CHECK-DAG:   [[TMP21:%.+]] = bitcast i32* [[ARGC_ADDR]] to i8*
// CHECK-DAG:   store i8* [[TMP21]], i8** [[TMP20:%[^,]+]],
// CHECK-DAG:   [[TMP20]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPARGC:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[TMP21:%.+]] = bitcast i32* [[ARGC_ADDR]] to i8*
// CHECK-DAG:   store i8* [[TMP21]], i8** [[TMP20:%[^,]+]],
// CHECK-DAG:   [[TMP20]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPARGC]], i32 0, i32 1
// CHECK-DAG:   [[GEPARGC]] = getelementptr inbounds [3 x [[T1]]], [3 x [[T1]]]* [[RD_IN1]], i64 0, i64
// CHECK-DAG:   [[TMP22:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPARGC]], i32 0, i32 2
// CHECK-DAG:   store i64 4, i64* [[TMP22]],
// CHECK-DAG:   [[TMP23:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPARGC]], i32 0, i32 3
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[ARGCINIT:.+]] to i8*), i8** [[TMP23]],
// CHECK-DAG:   [[TMP24:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPARGC]], i32 0, i32 4
// CHECK-DAG:   store i8* null, i8** [[TMP24]],
// CHECK-DAG:   [[TMP25:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPARGC]], i32 0, i32 5
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[ARGCCOMB:.+]] to i8*), i8** [[TMP25]],
// CHECK-DAG:   [[TMP26:%.+]] = getelementptr inbounds [[T1]], [[T1]]* [[GEPARGC]], i32 0, i32 6
// CHECK-DAG:   [[TMP27:%.+]] = bitcast i32* [[TMP26]] to i8*
// CHECK-DAG:   call void @llvm.memset.p0i8.i64(i8* align 8 [[TMP27]], i8 0, i64 4, i1 false)
// CHECK-DAG:   [[TMP28:%.+]] = bitcast [3 x [[T1]]]* [[RD_IN1]] to i8*
// CHECK-DAG:   [[TMP29:%.+]] = call i8* @__kmpc_taskred_init(i32 [[GTID]], i32 3, i8* [[TMP28]])
// DEBUG-DAG:   call void @llvm.dbg.declare(metadata i8** [[TD1]],
// CHECK-DAG:   store i8* [[TMP29]], i8** [[TD1]],
// CHECK-DAG:   call void @__kmpc_taskgroup(%struct.ident_t* {{[^,]+}}, i32 [[GTID]])
// CHECK-DAG:   [[TMP31:%.+]] = bitcast [5 x %struct.S]* [[C]] to i8*
// CHECK-DAG:   store i8* [[TMP31]], i8** [[TMP30:%[^,]+]],
// CHECK-DAG:   [[TMP30]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPC:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[TMP31:%.+]] = bitcast [5 x %struct.S]* [[C]] to i8*
// CHECK-DAG:   store i8* [[TMP31]], i8** [[TMP30:%[^,]+]],
// CHECK-DAG:   [[TMP30]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPC]], i32 0, i32 1
// CHECK-DAG:   [[GEPC]] = getelementptr inbounds [2 x [[T2]]], [2 x [[T2]]]* [[RD_IN2]], i64 0, i64
// CHECK-DAG:   [[TMP32:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPC]], i32 0, i32 2
// CHECK-DAG:   store i64 20, i64* [[TMP32]],
// CHECK-DAG:   [[TMP33:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPC]], i32 0, i32 3
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[CINIT:.+]] to i8*), i8** [[TMP33]],
// CHECK-DAG:   [[TMP34:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPC]], i32 0, i32 4
// CHECK-DAG:   store i8* bitcast (void (i8*)* @[[CFINI:.+]] to i8*), i8** [[TMP34]],
// CHECK-DAG:   [[TMP35:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPC]], i32 0, i32 5
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[CCOMB:.+]] to i8*), i8** [[TMP35]],
// CHECK-DAG:   [[TMP36:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPC]], i32 0, i32 6
// CHECK-DAG:   [[TMP37:%.+]] = bitcast i32* [[TMP36]] to i8*
// CHECK-DAG:   call void @llvm.memset.p0i8.i64(i8* align 8 [[TMP37]], i8 0, i64 4, i1 false)
// CHECK-DAG:   [[TMP39:%.+]] = bitcast i16* [[VLA]] to i8*
// CHECK-DAG:   store i8* [[TMP39]], i8** [[TMP38:%[^,]+]],
// CHECK-DAG:   [[TMP38]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPVLA:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[TMP39:%.+]] = bitcast i16* [[VLA]] to i8*
// CHECK-DAG:   store i8* [[TMP39]], i8** [[TMP38:%[^,]+]],
// CHECK-DAG:   [[TMP38]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPVLA]], i32 0, i32 1
// CHECK-DAG:   [[GEPVLA]] = getelementptr inbounds [2 x [[T2]]], [2 x [[T2]]]* [[RD_IN2]], i64 0, i64
// CHECK-DAG:   [[TMP40:%.+]] = mul nuw i64 [[VLA_SIZE]], 2
// CHECK-DAG:   [[TMP41:%.+]] = udiv exact i64 [[TMP40]], ptrtoint (i16* getelementptr (i16, i16* null, i32 1) to i64)
// CHECK-DAG:   [[TMP42:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPVLA]], i32 0, i32 2
// CHECK-DAG:   store i64 [[TMP40]], i64* [[TMP42]],
// CHECK-DAG:   [[TMP43:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPVLA]], i32 0, i32 3
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[VLAINIT:.+]] to i8*), i8** [[TMP43]],
// CHECK-DAG:   [[TMP44:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPVLA]], i32 0, i32 4
// CHECK-DAG:   store i8* null, i8** [[TMP44]],
// CHECK-DAG:   [[TMP45:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPVLA]], i32 0, i32 5
// CHECK-DAG:   store i8* bitcast (void (i8*, i8*)* @[[VLACOMB:.+]] to i8*), i8** [[TMP45]],
// CHECK-DAG:   [[TMP46:%.+]] = getelementptr inbounds [[T2]], [[T2]]* [[GEPVLA]], i32 0, i32 6
// CHECK-DAG:   store i32 1, i32* [[TMP46]],
// CHECK:       [[TMP47:%.+]] = bitcast [2 x [[T2]]]* [[RD_IN2]] to i8*
// CHECK:       [[TMP48:%.+]] = call i8* @__kmpc_taskred_init(i32 [[GTID]], i32 2, i8* [[TMP47]])
// CHECK:       store i8* [[TMP48]], i8** [[TD2]],
// CHECK:       call void @__kmpc_end_taskgroup(%struct.ident_t* {{[^,]+}}, i32 [[GTID]])
// CHECK:       call void @__kmpc_end_taskgroup(%struct.ident_t* {{[^,]+}}, i32 [[GTID]])

// CHECK-DAG: define internal void @[[AINIT]](i8* noalias %{{.+}}, i8* noalias %{{.+}})
// CHECK-DAG: store i32 0, i32* %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[ACOMB]](i8* %0, i8* %1)
// CHECK-DAG: add nsw i32 %
// CHECK-DAG: store i32 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[BINIT]](i8* noalias %{{.+}}, i8* noalias %{{.+}})
// CHECK-DAG: store float 0.000000e+00, float* %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[BCOMB]](i8* %0, i8* %1)
// CHECK-DAG: fadd float %
// CHECK-DAG: store float %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[ARGCINIT]](i8* noalias %{{.+}}, i8* noalias %{{.+}})
// CHECK-DAG: store i32 0, i32* %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[ARGCCOMB]](i8* %0, i8* %1)
// CHECK-DAG: add nsw i32 %
// CHECK-DAG: store i32 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[CINIT]](i8* noalias %{{.+}}, i8* noalias %{{.+}})
// CHECK-DAG: phi %struct.S* [
// CHECK-DAG: call {{.+}}(%struct.S* {{.+}})
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[CFINI]](i8* %0)
// CHECK-DAG: phi %struct.S* [
// CHECK-DAG: call {{.+}}(%struct.S* {{.+}})
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[CCOMB]](i8* %0, i8* %1)
// CHECK-DAG: phi %struct.S* [
// CHECK-DAG: phi %struct.S* [
// CHECK-DAG: call {{.+}}(%struct.S* {{.+}}, %struct.S* {{.+}}, %struct.S* {{.+}})
// CHECK-DAG: call {{.+}}(%struct.S* {{.+}}, %struct.S* {{.+}})
// CHECK-DAG: call {{.+}}(%struct.S* {{.+}})
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK_DAG: }

// CHECK-DAG: define internal void @[[VLAINIT]](i8* noalias %{{.+}}, i8* noalias %{{.+}})
// CHECK-DAG: call i32 @__kmpc_global_thread_num(%struct.ident_t* {{[^,]+}})
// CHECK-DAG: call i8* @__kmpc_threadprivate_cached(%struct.ident_t*
// CHECK-DAG: phi i16* [
// CHECK-DAG: store i16 0, i16* %
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[VLACOMB]](i8* %0, i8* %1)
// CHECK-DAG: call i32 @__kmpc_global_thread_num(%struct.ident_t* {{[^,]+}})
// CHECK-DAG: call i8* @__kmpc_threadprivate_cached(%struct.ident_t*
// CHECK-DAG: phi i16* [
// CHECK-DAG: phi i16* [
// CHECK-DAG: sext i16 %{{.+}} to i32
// CHECK-DAG: add nsw i32 %
// CHECK-DAG: trunc i32 %{{.+}} to i16
// CHECK-DAG: store i16 %
// CHECK_DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }
#endif

// DEBUG-LABEL: distinct !DICompileUnit
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[AINIT]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[ACOMB]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[BINIT]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[BCOMB]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[ARGCINIT]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[ARGCCOMB]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[CINIT]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[CFINI]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[CCOMB]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[VLAINIT]]",
// DEBUG-DAG: distinct !DISubprogram(linkageName: "[[VLACOMB]]",
