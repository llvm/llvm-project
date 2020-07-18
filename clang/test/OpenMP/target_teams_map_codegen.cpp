// Test host codegen.
// RUN: %clang_cc1 -DHAS_INT128 -verify -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefixes=CHECK,HOST,INT128,HOST-INT128
// RUN: %clang_cc1 -DHAS_INT128 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -DHAS_INT128 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,HOST,INT128,HOST-INT128
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s -check-prefixes=CHECK,HOST
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,HOST

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -DHAS_INT128 -verify -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DHAS_INT128 -verify -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s -check-prefixes=CHECK,INT128
// RUN: %clang_cc1 -DHAS_INT128 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -DHAS_INT128 -fopenmp -fopenmp-version=50 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK,INT128
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// HOST: @[[MAPTYPES_PRIVATE:.offload_maptypes[0-9.]*]] = private {{.*}}constant [2 x i64] [i64 35, i64 35]
// HOST: @[[MAPTYPES_FIRSTPRIVATE:.offload_maptypes[0-9.]*]] = private {{.*}}constant [2 x i64] [i64 35, i64 35]
// HOST: @[[MAPTYPES_REDUCTION:.offload_maptypes[0-9.]*]] = private {{.*}}constant [2 x i64] [i64 35, i64 35]
// HOST: @[[MAPTYPES_FROM:.offload_maptypes[0-9.]*]] = private {{.*}}constant [1 x i64] [i64 34]
// HOST: @[[MAPTYPES_TO:.offload_maptypes[0-9.]*]] = private {{.*}}constant [1 x i64] [i64 33]
// HOST: @[[MAPTYPES_ALLOC:.offload_maptypes[0-9.]*]] = private {{.*}}constant [1 x i64] [i64 32]
// HOST: @[[MAPTYPES_ARRAY_R0:.offload_maptypes[0-9.]*]] = private {{.*}}constant [3 x i64] [i64 35, i64 35, i64 35]
// HOST: @[[MAPTYPES_ARRAY_R1:.offload_maptypes[0-9.]*]] = private {{.*}}constant [3 x i64] [i64 33, i64 33, i64 33]
// HOST-INT128: @[[MAPTYPES_INT128_R0:.offload_maptypes[0-9.]*]] = private {{.*}}constant [3 x i64] [i64 35, i64 35, i64 35]
// HOST-INT128: @[[MAPTYPES_INT128_R1:.offload_maptypes[0-9.]*]] = private {{.*}}constant [3 x i64] [i64 34, i64 34, i64 34]
//
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_PRIVATE:__omp_offloading_[^"\\]*mapWithPrivate[^"\\]*]]\00"
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_FIRSTPRIVATE:__omp_offloading_[^"\\]*mapWithFirstprivate[^"\\]*]]\00"
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_REDUCTION:__omp_offloading_[^"\\]*mapWithReduction[^"\\]*]]\00"
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_FROM:__omp_offloading_[^"\\]*mapFrom[^"\\]*]]\00"
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_TO:__omp_offloading_[^"\\]*mapTo[^"\\]*]]\00"
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_ALLOC:__omp_offloading_[^"\\]*mapAlloc[^"\\]*]]\00"
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_ARRAY_R0:__omp_offloading_[^"\\]*mapArray[^"\\]*]]\00"
// CHECK: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_ARRAY_R1:__omp_offloading_[^"\\]*mapArray[^"\\]*]]\00"
// INT128: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_INT128_R0:__omp_offloading_[^"\\]*mapInt128[^"\\]*]]\00"
// INT128: @.omp_offloading.entry_name{{[0-9.]*}} = {{.*}} c"[[OFFLOAD_INT128_R1:__omp_offloading_[^"\\]*mapInt128[^"\\]*]]\00"

// HOST: define {{.*}}mapWithPrivate
// HOST: call {{.*}} @.[[OFFLOAD_PRIVATE]].region_id{{.*}} @[[MAPTYPES_PRIVATE]]
//
// CHECK: define {{.*}} void @[[OFFLOAD_PRIVATE]]()
// CHECK: call void ({{.*}}@[[OUTLINE_PRIVATE:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_PRIVATE]]({{.*}} %.global_tid., {{.*}} %.bound_tid.)
void mapWithPrivate() {
  int x, y;
  #pragma omp target teams private(x) map(x,y) private(y)
    ;
}

// HOST: define {{.*}}mapWithFirstprivate
// HOST: call {{.*}} @.[[OFFLOAD_FIRSTPRIVATE]].region_id{{.*}} @[[MAPTYPES_FIRSTPRIVATE]]
//
// CHECK: define {{.*}} void @[[OFFLOAD_FIRSTPRIVATE]](i{{[0-9]*}}* {{[^,]*}}%x, i{{[0-9]*}}* {{[^,]*}}%y)
// CHECK: call void ({{.*}}@[[OUTLINE_FIRSTPRIVATE:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_FIRSTPRIVATE]]({{.*}} %.global_tid., {{.*}} %.bound_tid., i{{[0-9]*}} %x, i{{[0-9]*}} %y)
void mapWithFirstprivate() {
  int x, y;
  #pragma omp target teams firstprivate(x) map(x,y) firstprivate(y)
    ;
}

// HOST: define {{.*}}mapWithReduction
// HOST: call {{.*}} @.[[OFFLOAD_REDUCTION]].region_id{{.*}} @[[MAPTYPES_REDUCTION]]
//
// CHECK: define {{.*}} void @[[OFFLOAD_REDUCTION]](i{{[0-9]*}}* {{[^,]*}}%x, i{{[0-9]*}}* {{[^,]*}}%y)
// CHECK: call void ({{.*}}@[[OUTLINE_REDUCTION:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_REDUCTION]]({{[^)]*}} i{{[0-9]*}}* {{[^,]*}}%x, i{{[0-9]*}}* {{[^,]*}}%y)
// CHECK: %.omp.reduction.red_list = alloca [2 x i8*]
void mapWithReduction() {
  int x, y;
  #pragma omp target teams reduction(+:x) map(x,y) reduction(+:y)
    ;
}

// HOST: define {{.*}}mapFrom
// HOST: call {{.*}} @.[[OFFLOAD_FROM]].region_id{{.*}} @[[MAPTYPES_FROM]]
//
// CHECK: define {{.*}} void @[[OFFLOAD_FROM]](i{{[0-9]*}}* {{[^,]*}}%x)
// CHECK: call void ({{.*}}@[[OUTLINE_FROM:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_FROM]]({{.*}} %.global_tid., {{.*}} %.bound_tid., i{{[0-9]*}} %x)
void mapFrom() {
  int x;
  #pragma omp target teams firstprivate(x) map(from:x)
    ;
}

// HOST: define {{.*}}mapTo
// HOST: call {{.*}} @.[[OFFLOAD_TO]].region_id{{.*}} @[[MAPTYPES_TO]]
//
// CHECK: define {{.*}} void @[[OFFLOAD_TO]](i{{[0-9]*}}* {{[^,]*}}%x)
// CHECK: call void ({{.*}}@[[OUTLINE_TO:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_TO]]({{.*}} %.global_tid., {{.*}} %.bound_tid., i{{[0-9]*}} %x)
void mapTo() {
  int x;
  #pragma omp target teams firstprivate(x) map(to:x)
    ;
}

// HOST: define {{.*}}mapAlloc
// HOST: call {{.*}} @.[[OFFLOAD_ALLOC]].region_id{{.*}} @[[MAPTYPES_ALLOC]]
//
// CHECK: define {{.*}} void @[[OFFLOAD_ALLOC]](i{{[0-9]*}}* {{[^,]*}}%x)
// CHECK: call void ({{.*}}@[[OUTLINE_ALLOC:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_ALLOC]]({{.*}} %.global_tid., {{.*}} %.bound_tid., i{{[0-9]*}} %x)
void mapAlloc() {
  int x;
  #pragma omp target teams firstprivate(x) map(alloc:x)
    ;
}

// HOST: define {{.*}}mapArray
// HOST: call {{.*}} @.[[OFFLOAD_ARRAY_R0]].region_id{{.*}} @[[MAPTYPES_ARRAY_R0]]
// HOST: call {{.*}} @.[[OFFLOAD_ARRAY_R1]].region_id{{.*}} @[[MAPTYPES_ARRAY_R1]]
//
// CHECK: define {{.*}} void @[[OFFLOAD_ARRAY_R0]]([88 x i{{[0-9]*}}]* {{[^,]*}}%y, [99 x i{{[0-9]*}}]* {{[^,]*}}%z)
// CHECK: call void ({{.*}}@[[OUTLINE_ARRAY_R0:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_ARRAY_R0]]({{.*}} %.global_tid., {{.*}} %.bound_tid., [88 x i{{[0-9]*}}]* {{[^,]*}}%y, [99 x i{{[0-9]*}}]* {{[^,]*}}%z)
// CHECK: %.omp.reduction.red_list = alloca [1 x i8*]
//
// CHECK: define {{.*}} void @[[OFFLOAD_ARRAY_R1]]([88 x i{{[0-9]*}}]* {{[^,]*}}%y, [99 x i{{[0-9]*}}]* {{[^,]*}}%z)
// CHECK: call void ({{.*}}@[[OUTLINE_ARRAY_R1:.omp_outlined.[.0-9]*]]
//
// CHECK: define {{.*}} void @[[OUTLINE_ARRAY_R1]]({{.*}} %.global_tid., {{.*}} %.bound_tid., [88 x i{{[0-9]*}}]* {{[^,]*}}%y, [99 x i{{[0-9]*}}]* {{[^,]*}}%z)
// CHECK: %.omp.reduction.red_list = alloca [1 x i8*]
void mapArray() {
  int x[77], y[88], z[99];
  #pragma omp target teams private(x) firstprivate(y) reduction(+:z) map(x,y,z)
    ;
  #pragma omp target teams private(x) firstprivate(y) reduction(+:z) map(to:x,y,z)
    ;
}

# if HAS_INT128
// HOST-INT128: define {{.*}}mapInt128
// HOST-INT128: call {{.*}} @.[[OFFLOAD_INT128_R0]].region_id{{.*}} @[[MAPTYPES_INT128_R0]]
// HOST-INT128: call {{.*}} @.[[OFFLOAD_INT128_R1]].region_id{{.*}} @[[MAPTYPES_INT128_R1]]
//
// INT128: define {{.*}} void @[[OFFLOAD_INT128_R0]](i128* {{[^,]*}}%y, i128* {{[^,]*}}%z)
// INT128: call void ({{.*}}@[[OUTLINE_INT128_R0:.omp_outlined.[.0-9]*]]
//
// INT128: define {{.*}} void @[[OUTLINE_INT128_R0]]({{.*}} %.global_tid., {{.*}} %.bound_tid., i128* {{[^,]*}}%y, i128* {{[^,]*}}%z)
// INT128: %.omp.reduction.red_list = alloca [1 x i8*]
//
// INT128: define {{.*}} void @[[OFFLOAD_INT128_R1]](i128* {{[^,]*}}%y, i128* {{[^,]*}}%z)
// INT128: call void ({{.*}}@[[OUTLINE_INT128_R1:.omp_outlined.[.0-9]*]]
//
// INT128: define {{.*}} void @[[OUTLINE_INT128_R1]]({{.*}} %.global_tid., {{.*}} %.bound_tid., i128* {{[^,]*}}%y, i128* {{[^,]*}}%z)
// INT128: %.omp.reduction.red_list = alloca [1 x i8*]
void mapInt128() {
  __int128 x, y, z;
  #pragma omp target teams private(x) firstprivate(y) reduction(+:z) map(x,y,z)
    ;
  #pragma omp target teams private(x) firstprivate(y) reduction(+:z) map(from:x,y,z)
    ;
}
# endif
#endif
