//RUN: %clang_cc1 -verify -x c++ -triple x86_64 -fopenmp -fopenmp-version=51 \
//RUN:  -fopenmp-targets=x86_64 -I%S/Inputs -emit-llvm -o - %s | FileCheck %s

//RUN: %clang_cc1 -x c++ -triple x86_64 -fopenmp -fopenmp-version=51 \
//RUN:  -fopenmp-targets=x86_64 -I%S/Inputs -emit-llvm-bc -o %t-host.bc %s

//RUN: %clang_cc1 -x c++ -triple x86_64 -fopenmp -fopenmp-version=51 \
//RUN:  -fopenmp-targets=x86_64 -I%S/Inputs -fopenmp-is-target-device \
//RUN:  -fopenmp-host-ir-file-path %t-host.bc -emit-llvm -o - %s \
//RUN:  | FileCheck %s --check-prefix=TCHECK

// expected-no-diagnostics

//CHECK: define {{.*}}void @[[FOO:.+]](
void foo() {
  int i = 0;

//CHECK: call void @__omp_offloading_[[FILEID1:[0-9a-f]+_[0-9a-f]+]]_[[FOO]]_l[[T1L:[0-9]+]](

#define VALUE 1
#include "multiple_regions.inc"

//CHECK: call void @__omp_offloading_[[FILEID1]]_[[FOO]]_l[[T1L]]_1(
#undef VALUE
#define VALUE 2
#include "multiple_regions.inc"

//CHECK: call void @__omp_offloading_[[FILEID1]]_[[FOO]]_l[[T1L]]_2(
#undef VALUE
#define VALUE 3
#include "multiple_regions.inc"
}

//CHECK: define {{.*}}void @__omp_offloading_[[FILEID1]]_[[FOO]]_l[[T1L]](
//CHECK: define {{.*}}void @__omp_offloading_[[FILEID1]]_[[FOO]]_l[[T1L]]_1(
//CHECK: define {{.*}}void @__omp_offloading_[[FILEID1]]_[[FOO]]_l[[T1L]]_2(
//TCHECK: define {{.*}}void @__omp_offloading_[[FILEID1:[0-9a-f]+_[0-9a-f]+]]_[[FOO:.+]]_l[[T1L:[0-9]+]](
//TCHECK: define {{.*}}void @__omp_offloading_[[FILEID1]]_[[FOO]]_l[[T1L]]_1(
//TCHECK: define {{.*}}void @__omp_offloading_[[FILEID1]]_[[FOO]]_l[[T1L]]_2(

#define A()\
_Pragma("omp target")\
{}\
_Pragma("omp target")\
{}

//CHECK: define {{.*}}void @[[BAR:.+]](
void bar()
{
//CHECK: call void @__omp_offloading_[[FILEID2:[0-9a-f]+_[0-9a-f]+]]_[[BAR]]_l[[T2L:[0-9]+]](
//CHECK: call void @__omp_offloading_[[FILEID2]]_[[BAR]]_l[[T2L]]_1(
  A()
}

//CHECK: define {{.*}}void @__omp_offloading_[[FILEID2]]_[[BAR]]_l[[T2L]](
//CHECK: define {{.*}}void @__omp_offloading_[[FILEID2]]_[[BAR]]_l[[T2L]]_1(
//TCHECK: define {{.*}}void @__omp_offloading_[[FILEID2:[0-9a-f]+_[0-9a-f]+]]_[[BAR:.+]]_l[[T2L:[0-9]+]](
//TCHECK: define {{.*}}void @__omp_offloading_[[FILEID2]]_[[BAR]]_l[[T2L]]_1(

// Check metadata is properly generated:
// CHECK:     !omp_offload.info = !{!{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[FOO]]", i32 [[T1L]], i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[FOO]]", i32 [[T1L]], i32 1, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[FOO]]", i32 [[T1L]], i32 2, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[BAR]]", i32 [[T2L]], i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[BAR]]", i32 [[T2L]], i32 1, i32 {{[0-9]+}}}

// TCHECK:     !omp_offload.info = !{!{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[FOO]]", i32 [[T1L]], i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[FOO]]", i32 [[T1L]], i32 1, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[FOO]]", i32 [[T1L]], i32 2, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[BAR]]", i32 [[T2L]], i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 {{-?[0-9]+}}, i32 {{-?[0-9]+}}, !"[[BAR]]", i32 [[T2L]], i32 1, i32 {{[0-9]+}}}
