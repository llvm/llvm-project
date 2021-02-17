// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// SIMD-ONLY30-NOT: {{__kmpc|__tgt}}
#ifdef CK30

// CK30-DAG: [[BASE:%.+]] = type { i32*, i32, i32* }
// CK30-DAG: [[STRUCT:%.+]] = type { [[BASE]], i32*, i32*, i32, i32* }

// CK30-LABEL: @.__omp_offloading_{{.*}}map_with_deep_copy{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// The first element: 0x20 - OMP_MAP_TARGET_PARAM
// 2-4: 0x1000000000003 - OMP_MAP_MEMBER_OF(0) | OMP_MAP_TO | OMP_MAP_FROM - copies all the data in structs excluding deep-copied elements (from &s to &s.ptrBase1, from &s.ptr to &s.ptr1, from &s.ptr1 to end of s).
// 5-6: 0x1000000000013 - OMP_MAP_MEMBER_OF(0) | OMP_MAP_PTR_AND_OBJ | OMP_MAP_TO | OMP_MAP_FROM - deep copy of the pointers + pointee.
// CK30: [[MTYPE00:@.+]] = private {{.*}}constant [6 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710675, i64 281474976710675]

typedef struct {
  int *ptrBase;
  int valBase;
  int *ptrBase1;
} Base;

typedef struct StructWithPtrTag : public Base {
  int *ptr;
  int *ptr2;
  int val;
  int *ptr1;
} StructWithPtr;

// CK30-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @.__omp_offloading_{{.*}}map_with_deep_copy{{.*}}_l{{[0-9]+}}.region_id, i32 6, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], i64* getelementptr inbounds ([6 x i64], [6 x i64]* [[MTYPE00]], i32 0, i32 0), i8** null, i8** null)
// CK30-DAG: [[GEPS]] = getelementptr inbounds [6 x i{{64|32}}], [6 x i64]* [[SIZES:%.+]], i32 0, i32 0
// CK30-DAG: [[GEPP]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS:%.+]], i32 0, i32 0
// CK30-DAG: [[GEPBP]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES:%.+]], i32 0, i32 0

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 0
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S:%.+]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 0
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i{{64|32}}], [6 x i64]* [[SIZES]], i32 0, i32 0
// CK30-DAG: store i64 [[S_ALLOC_SIZE:%.+]], i64* [[SIZE]],
// CK30-DAG: [[S_ALLOC_SIZE]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_END_BC:%.+]], [[S_BEGIN_BC:%.+]]
// CK30-DAG: [[S_BEGIN_BC]] = ptrtoint i8* [[S_BEGIN:%.+]] to i64
// CK30-DAG: [[S_END_BC]] = ptrtoint i8* [[S_END:%.+]] to i64
// CK30-DAG: [[S_BEGIN]] = bitcast [[STRUCT]]* [[S]] to i8*
// CK30-DAG: [[S_END]] = bitcast [[STRUCT]]* [[REAL_S_END:%.+]] to i8*
// CK30-DAG: [[REAL_S_END]] = getelementptr [[STRUCT]], [[STRUCT]]* [[S]], i32 1

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 1
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 1
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 1
// CK30-DAG: store i64 [[SIZE1:%.+]], i64* [[SIZE]],
// CK30-DAG: [[SIZE1]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_PTRBASE1_BC:%.+]], [[S_BEGIN_BC:%.+]]
// CK30-DAG: [[S_BEGIN_BC]] = ptrtoint i8* [[S_BEGIN:%.+]] to i64
// CK30-DAG: [[S_PTRBASE1_BC]] = ptrtoint i8* [[S_PTRBASE1:%.+]] to i64
// CK30-DAG: [[S_PTRBASE1]] = bitcast i32** [[S_PTRBASE1_REF:%.+]] to i8*
// CK30-DAG: [[S_BEGIN]] = bitcast [[STRUCT]]* [[S]] to i8*
// CK30-DAG: [[S_PTRBASE1_REF]] = getelementptr inbounds [[BASE]], [[BASE]]* [[BASE_ADDR:%.+]], i32 0, i32 2
// CK30-DAG: [[BASE_ADDR]] = bitcast [[STRUCT]]* [[S]] to [[BASE]]*

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 2
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 2
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32***
// CK30-DAG: store i32** [[PTR1:%.+]], i32*** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 2
// CK30-DAG: store i64 [[SIZE2:%.+]], i64* [[SIZE]],
// CK30-DAG: [[PTR1]] = getelementptr i32*, i32** [[S_PTRBASE1_REF]], i{{64|32}} 1
// CK30-DAG: [[SIZE2]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_PTR1_BC:%.+]], [[S_PTRBASE1_BC:%.+]]
// CK30-DAG: [[S_PTR1_BC]] = ptrtoint i8* [[S_PTR1:%.+]] to i64
// CK30-DAG: [[S_PTRBASE1_BC]] = ptrtoint i8* [[S_PTRBASE1:%.+]] to i64
// CK30-DAG: [[S_PTR1]] = bitcast i32** [[S_PTR1_REF:%.+]] to i8*
// CK30-DAG: [[S_PTRBASE1]] = bitcast i32** [[PTR1]] to i8*
// CK30-DAG: [[S_PTR1_REF]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 4

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 3
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 3
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32***
// CK30-DAG: store i32** [[PTR2:%.+]], i32*** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 3
// CK30-DAG: store i64 [[SIZE3:%.+]], i64* [[SIZE]],
// CK30-DAG: [[PTR2]] = getelementptr i32*, i32** [[S_PTR1_REF]], i{{64|32}} 1
// CK30-DAG: [[SIZE3]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_END_BC:%.+]], [[S_PTR1_BC:%.+]]
// CK30-DAG: [[S_PTR1_BC]] = ptrtoint i8* [[S_PTR1:%.+]] to i64
// CK30-DAG: [[S_END_BC]] = ptrtoint i8* [[S_END:%.+]] to i64
// CK30-DAG: [[S_PTR1]] = bitcast i32** [[PTR2]] to i8*
// CK30-DAG: [[S_END]] = getelementptr i8, i8* [[S_LAST:%.+]], i{{64|32}} 1
// CK30-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_BC:%.+]], i{{64|32}} {{55|27}}
// CK30-DAG: [[S_BC]] = bitcast [[STRUCT]]* [[S]] to i8*

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 4
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to i32***
// CK30-DAG: store i32** [[S_PTR1:%.+]], i32*** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 4
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32**
// CK30-DAG: store i32* [[S_PTR1_BEGIN:%.+]], i32** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 4
// CK30-DAG: store i64 4, i64* [[SIZE]],
// CK30-DAG: [[S_PTR1]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 4
// CK30-DAG: [[S_PTR1_BEGIN]] = getelementptr inbounds i32, i32* [[S_PTR1_BEGIN_REF:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTR1_BEGIN_REF]] = load i32*, i32** [[S_PTR1:%.+]],
// CK30-DAG: [[S_PTR1]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 4

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 5
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to i32***
// CK30-DAG: store i32** [[S_PTRBASE1:%.+]], i32*** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 5
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32**
// CK30-DAG: store i32* [[S_PTRBASE1_BEGIN:%.+]], i32** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i{{64|32}}], [6 x i{{64|32}}]* [[SIZES]], i32 0, i32 5
// CK30-DAG: store i{{64|32}} 4, i{{64|32}}* [[SIZE]],
// CK30-DAG: [[S_PTRBASE1]] = getelementptr inbounds [[BASE]], [[BASE]]* [[S_BASE:%.+]], i32 0, i32 2
// CK30-DAG: [[S_BASE]] = bitcast [[STRUCT]]* [[S]] to [[BASE]]*
// CK30-DAG: [[S_PTRBASE1_BEGIN]] = getelementptr inbounds i32, i32* [[S_PTRBASE1_BEGIN_REF:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTRBASE1_BEGIN_REF]] = load i32*, i32** [[S_PTRBASE1:%.+]],
// CK30-DAG: [[S_PTRBASE1]] = getelementptr inbounds [[BASE]], [[BASE]]* [[S_BASE:%.+]], i32 0, i32 2
// CK30-DAG: [[S_BASE]] = bitcast [[STRUCT]]* [[S]] to [[BASE]]*
void map_with_deep_copy() {
  StructWithPtr s;
#pragma omp target map(s, s.ptr1 [0:1], s.ptrBase1 [0:1])
  {
    s.val++;
    s.ptr1[0]++;
    s.ptrBase1[0] = 10001;
  }
}

#endif // CK30
#endif
