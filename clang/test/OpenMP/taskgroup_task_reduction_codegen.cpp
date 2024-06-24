// RUN: %clang_cc1 -verify -Wno-vla -triple x86_64-apple-darwin10 -fopenmp -x c++ -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ %s -verify -Wno-vla -debug-info-kind=limited -emit-llvm -o - -triple x86_64-apple-darwin10 | FileCheck %s --check-prefix=CHECK --check-prefix=DEBUG

// RUN: %clang_cc1 -verify -Wno-vla -triple x86_64-apple-darwin10 -fopenmp-simd -x c++ -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ %s -verify -Wno-vla -debug-info-kind=limited -emit-llvm -o - -triple x86_64-apple-darwin10 | FileCheck --check-prefix SIMD-ONLY0 %s
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
// CHECK:       [[ARGV_ADDR:%.+]] = alloca ptr,
// CHECK:       [[A:%.+]] = alloca i32,
// CHECK:       [[B:%.+]] = alloca float,
// CHECK:       [[C:%.+]] = alloca [5 x %struct.S],
// CHECK:       [[RD_IN1:%.+]] = alloca [3 x [[T1:%[^,]+]]],
// CHECK:       [[TD1:%.+]] = alloca ptr,
// CHECK:       [[RD_IN2:%.+]] = alloca [2 x [[T2:%[^,]+]]],
// CHECK:       [[TD2:%.+]] = alloca ptr,

// CHECK:       [[GTID:%.+]] = call i32 @__kmpc_global_thread_num(ptr
// CHECK:       [[VLA:%.+]] = alloca i16, i64 [[VLA_SIZE:%[^,]+]],

// CHECK:       call void @__kmpc_taskgroup(ptr {{[^,]+}}, i32 [[GTID]])
// CHECK-DAG:   store ptr [[A]], ptr [[A_REF:[^,]+]],
// CHECK-DAG:   [[A_REF]] = getelementptr inbounds [[T1]], ptr [[GEPA:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   store ptr [[A]], ptr [[A_REF:[^,]+]],
// CHECK-DAG:   [[A_REF]] = getelementptr inbounds [[T1]], ptr [[GEPA]], i32 0, i32 1
// CHECK-DAG:   [[GEPA]] = getelementptr inbounds [3 x [[T1]]], ptr [[RD_IN1]], i64 0, i64
// CHECK-DAG:   [[TMP6:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPA]], i32 0, i32 2
// CHECK-DAG:   store i64 4, ptr [[TMP6]],
// CHECK-DAG:   [[TMP7:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPA]], i32 0, i32 3
// CHECK-DAG:   store ptr @[[AINIT:.+]], ptr [[TMP7]],
// CHECK-DAG:   [[TMP8:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPA]], i32 0, i32 4
// CHECK-DAG:   store ptr null, ptr [[TMP8]],
// CHECK-DAG:   [[TMP9:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPA]], i32 0, i32 5
// CHECK-DAG:   store ptr @[[ACOMB:.+]], ptr [[TMP9]],
// CHECK-DAG:   [[TMP10:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPA]], i32 0, i32 6
// CHECK-DAG:   call void @llvm.memset.p0.i64(ptr align 8 [[TMP10]], i8 0, i64 4, i1 false)
// CHECK-DAG:   store ptr [[B]], ptr [[TMP12:%[^,]+]],
// CHECK-DAG:   [[TMP12]] = getelementptr inbounds [[T1]], ptr [[GEPB:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   store ptr [[B]], ptr [[TMP12:%[^,]+]],
// CHECK-DAG:   [[TMP12]] = getelementptr inbounds [[T1]], ptr [[GEPB]], i32 0, i32 1
// CHECK-DAG:   [[GEPB]] = getelementptr inbounds [3 x [[T1]]], ptr [[RD_IN1]], i64 0, i64
// CHECK-DAG:   [[TMP14:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPB]], i32 0, i32 2
// CHECK-DAG:   store i64 4, ptr [[TMP14]],
// CHECK-DAG:   [[TMP15:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPB]], i32 0, i32 3
// CHECK-DAG:   store ptr @[[BINIT:.+]], ptr [[TMP15]],
// CHECK-DAG:   [[TMP16:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPB]], i32 0, i32 4
// CHECK-DAG:   store ptr null, ptr [[TMP16]],
// CHECK-DAG:   [[TMP17:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPB]], i32 0, i32 5
// CHECK-DAG:   store ptr @[[BCOMB:.+]], ptr [[TMP17]],
// CHECK-DAG:   [[TMP18:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPB]], i32 0, i32 6
// CHECK-DAG:   call void @llvm.memset.p0.i64(ptr align 8 [[TMP18]], i8 0, i64 4, i1 false)
// CHECK-DAG:   store ptr [[ARGC_ADDR]], ptr [[TMP20:%[^,]+]],
// CHECK-DAG:   [[TMP20]] = getelementptr inbounds [[T1]], ptr [[GEPARGC:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   store ptr [[ARGC_ADDR]], ptr [[TMP20:%[^,]+]],
// CHECK-DAG:   [[TMP20]] = getelementptr inbounds [[T1]], ptr [[GEPARGC]], i32 0, i32 1
// CHECK-DAG:   [[GEPARGC]] = getelementptr inbounds [3 x [[T1]]], ptr [[RD_IN1]], i64 0, i64
// CHECK-DAG:   [[TMP22:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPARGC]], i32 0, i32 2
// CHECK-DAG:   store i64 4, ptr [[TMP22]],
// CHECK-DAG:   [[TMP23:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPARGC]], i32 0, i32 3
// CHECK-DAG:   store ptr @[[ARGCINIT:.+]], ptr [[TMP23]],
// CHECK-DAG:   [[TMP24:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPARGC]], i32 0, i32 4
// CHECK-DAG:   store ptr null, ptr [[TMP24]],
// CHECK-DAG:   [[TMP25:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPARGC]], i32 0, i32 5
// CHECK-DAG:   store ptr @[[ARGCCOMB:.+]], ptr [[TMP25]],
// CHECK-DAG:   [[TMP26:%.+]] = getelementptr inbounds [[T1]], ptr [[GEPARGC]], i32 0, i32 6
// CHECK-DAG:   call void @llvm.memset.p0.i64(ptr align 8 [[TMP26]], i8 0, i64 4, i1 false)
// CHECK-DAG:   [[TMP29:%.+]] = call ptr @__kmpc_taskred_init(i32 [[GTID]], i32 3, ptr [[RD_IN1]])
// DEBUG-DAG:   #dbg_declare(ptr [[TD1]],
// CHECK-DAG:   store ptr [[TMP29]], ptr [[TD1]],
// CHECK-DAG:   call void @__kmpc_taskgroup(ptr {{[^,]+}}, i32 [[GTID]])
// CHECK-DAG:   store ptr [[C]], ptr [[TMP30:%[^,]+]],
// CHECK-DAG:   [[TMP30]] = getelementptr inbounds [[T2]], ptr [[GEPC:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   store ptr [[C]], ptr [[TMP30:%[^,]+]],
// CHECK-DAG:   [[TMP30]] = getelementptr inbounds [[T2]], ptr [[GEPC]], i32 0, i32 1
// CHECK-DAG:   [[GEPC]] = getelementptr inbounds [2 x [[T2]]], ptr [[RD_IN2]], i64 0, i64
// CHECK-DAG:   [[TMP32:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPC]], i32 0, i32 2
// CHECK-DAG:   store i64 20, ptr [[TMP32]],
// CHECK-DAG:   [[TMP33:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPC]], i32 0, i32 3
// CHECK-DAG:   store ptr @[[CINIT:.+]], ptr [[TMP33]],
// CHECK-DAG:   [[TMP34:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPC]], i32 0, i32 4
// CHECK-DAG:   store ptr @[[CFINI:.+]], ptr [[TMP34]],
// CHECK-DAG:   [[TMP35:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPC]], i32 0, i32 5
// CHECK-DAG:   store ptr @[[CCOMB:.+]], ptr [[TMP35]],
// CHECK-DAG:   [[TMP36:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPC]], i32 0, i32 6
// CHECK-DAG:   call void @llvm.memset.p0.i64(ptr align 8 [[TMP36]], i8 0, i64 4, i1 false)
// CHECK-DAG:   store ptr [[VLA]], ptr [[TMP38:%[^,]+]],
// CHECK-DAG:   [[TMP38]] = getelementptr inbounds [[T2]], ptr [[GEPVLA:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   store ptr [[VLA]], ptr [[TMP38:%[^,]+]],
// CHECK-DAG:   [[TMP38]] = getelementptr inbounds [[T2]], ptr [[GEPVLA]], i32 0, i32 1
// CHECK-DAG:   [[GEPVLA]] = getelementptr inbounds [2 x [[T2]]], ptr [[RD_IN2]], i64 0, i64
// CHECK-DAG:   [[TMP40:%.+]] = mul nuw i64 [[VLA_SIZE]], 2
// CHECK-DAG:   [[TMP41:%.+]] = udiv exact i64 [[TMP40]], ptrtoint (ptr getelementptr (i16, ptr null, i32 1) to i64)
// CHECK-DAG:   [[TMP42:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPVLA]], i32 0, i32 2
// CHECK-DAG:   store i64 [[TMP40]], ptr [[TMP42]],
// CHECK-DAG:   [[TMP43:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPVLA]], i32 0, i32 3
// CHECK-DAG:   store ptr @[[VLAINIT:.+]], ptr [[TMP43]],
// CHECK-DAG:   [[TMP44:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPVLA]], i32 0, i32 4
// CHECK-DAG:   store ptr null, ptr [[TMP44]],
// CHECK-DAG:   [[TMP45:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPVLA]], i32 0, i32 5
// CHECK-DAG:   store ptr @[[VLACOMB:.+]], ptr [[TMP45]],
// CHECK-DAG:   [[TMP46:%.+]] = getelementptr inbounds [[T2]], ptr [[GEPVLA]], i32 0, i32 6
// CHECK-DAG:   store i32 1, ptr [[TMP46]],
// CHECK:       [[TMP48:%.+]] = call ptr @__kmpc_taskred_init(i32 [[GTID]], i32 2, ptr [[RD_IN2]])
// CHECK:       store ptr [[TMP48]], ptr [[TD2]],
// CHECK:       call void @__kmpc_end_taskgroup(ptr {{[^,]+}}, i32 [[GTID]])
// CHECK:       call void @__kmpc_end_taskgroup(ptr {{[^,]+}}, i32 [[GTID]])

// CHECK-DAG: define internal void @[[AINIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-DAG: store i32 0, ptr %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[ACOMB]](ptr noundef %0, ptr noundef %1)
// CHECK-DAG: add nsw i32 %
// CHECK-DAG: store i32 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[BINIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-DAG: store float 0.000000e+00, ptr %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[BCOMB]](ptr noundef %0, ptr noundef %1)
// CHECK-DAG: fadd float %
// CHECK-DAG: store float %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[ARGCINIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-DAG: store i32 0, ptr %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[ARGCCOMB]](ptr noundef %0, ptr noundef %1)
// CHECK-DAG: add nsw i32 %
// CHECK-DAG: store i32 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[CINIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-DAG: phi ptr [
// CHECK-DAG: call {{.+}}(ptr {{.+}})
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[CFINI]](ptr noundef %0)
// CHECK-DAG: phi ptr [
// CHECK-DAG: call {{.+}}(ptr {{.+}})
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[CCOMB]](ptr noundef %0, ptr noundef %1)
// CHECK-DAG: phi ptr [
// CHECK-DAG: phi ptr [
// CHECK-DAG: call {{.+}}(ptr {{.+}}, ptr {{.+}}, ptr {{.+}})
// CHECK-DAG: call {{.+}}(ptr {{.+}}, ptr {{.+}})
// CHECK-DAG: call {{.+}}(ptr {{.+}})
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[VLAINIT]](ptr noalias noundef %{{.+}}, ptr noalias noundef %{{.+}})
// CHECK-DAG: call i32 @__kmpc_global_thread_num(ptr {{[^,]+}})
// CHECK-DAG: call ptr @__kmpc_threadprivate_cached(ptr
// CHECK-DAG: phi ptr [
// CHECK-DAG: store i16 0, ptr %
// CHECK-DAG: br i1 %
// CHECK-DAG: ret void
// CHECK-DAG: }

// CHECK-DAG: define internal void @[[VLACOMB]](ptr noundef %0, ptr noundef %1)
// CHECK-DAG: call i32 @__kmpc_global_thread_num(ptr {{[^,]+}})
// CHECK-DAG: call ptr @__kmpc_threadprivate_cached(ptr
// CHECK-DAG: phi ptr [
// CHECK-DAG: phi ptr [
// CHECK-DAG: sext i16 %{{.+}} to i32
// CHECK-DAG: add nsw i32 %
// CHECK-DAG: trunc i32 %{{.+}} to i16
// CHECK-DAG: store i16 %
// CHECK-DAG: br i1 %
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
