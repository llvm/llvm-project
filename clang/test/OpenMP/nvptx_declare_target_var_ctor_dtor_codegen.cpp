// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -o - | FileCheck %s --check-prefix HOST --check-prefix CHECK
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fvisibility=protected -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK
// RUN: %clang_cc1 -fopenmp -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-target-device -fvisibility=protected -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -fopenmp -x c++ -triple nvptx64-nvidia-cuda -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fvisibility=protected -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefix DEVICE --check-prefix CHECK

// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o - | FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fvisibility=protected -fopenmp-host-ir-file-path %t-ppc-host.bc -o -| FileCheck %s --check-prefix SIMD-ONLY
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda %s -fopenmp-is-target-device -fvisibility=protected -fopenmp-host-ir-file-path %t-ppc-host.bc -emit-pch -o %t
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fvisibility=protected -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -o - | FileCheck %s --check-prefix SIMD-ONLY

#ifndef HEADER
#define HEADER

// SIMD-ONLY-NOT: {{__kmpc|__tgt}}

// DEVICE-DAG: [[C_ADDR:.+]] = internal global i32 0,
// DEVICE-DAG: [[CD_ADDR:@.+]] ={{ protected | }}global %struct.S zeroinitializer,
// HOST-DAG: @[[C_ADDR:.+]] = internal global i32 0,
// HOST-DAG: @[[CD_ADDR:.+]] ={{( protected | dso_local)?}} global %struct.S zeroinitializer,

// DEVICE-DAG: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @[[CTOR:.+]], ptr null }]
// DEVICE-DAG: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @[[DTOR:.+]], ptr null }]

#pragma omp declare target
int foo() { return 0; }
#pragma omp end declare target
int bar() { return 0; }
#pragma omp declare target (bar)
int baz() { return 0; }

#pragma omp declare target
int doo() { return 0; }
#pragma omp end declare target
int car() { return 0; }
#pragma omp declare target (bar)
int caz() { return 0; }

// DEVICE-DAG: define hidden noundef i32 [[FOO:@.*foo.*]]()
// DEVICE-DAG: define hidden noundef i32 [[BAR:@.*bar.*]]()
// DEVICE-DAG: define hidden noundef i32 [[BAZ:@.*baz.*]]()
// DEVICE-DAG: define hidden noundef i32 [[DOO:@.*doo.*]]()
// DEVICE-DAG: define hidden noundef i32 [[CAR:@.*car.*]]()
// DEVICE-DAG: define hidden noundef i32 [[CAZ:@.*caz.*]]()

static int c = foo() + bar() + baz();
#pragma omp declare target (c)

struct S {
  int a;
  S() = default;
  S(int a) : a(a) {}
  ~S() { a = 0; }
};

#pragma omp declare target
S cd = doo() + car() + caz() + baz();
#pragma omp end declare target

int maini1() {
  int a;
#pragma omp target map(tofrom : a)
  {
    a = c;
  }
  return 0;
}

// DEVICE-DAG: define weak{{.*}} void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l[[@LINE-7]](ptr {{[^,]*}}, ptr noundef nonnull align {{[0-9]+}} dereferenceable{{[^,]*}}
// DEVICE-DAG: [[C:%.+]] = load i32, ptr [[C_ADDR]],
// DEVICE-DAG: store i32 [[C]], ptr %

// HOST: define internal void @__omp_offloading_{{.*}}_{{.*}}maini1{{.*}}_l[[@LINE-11]](ptr noundef nonnull align {{[0-9]+}} dereferenceable{{.*}})
// HOST: [[C:%.*]] = load i32, ptr @[[C_ADDR]],
// HOST: store i32 [[C]], ptr %

// HOST-DAG: !{i32 1, !"[[CD_ADDR]]", i32 0, i32 {{[0-9]+}}}
// HOST-DAG: !{i32 1, !"[[C_ADDR]]", i32 0, i32 {{[0-9]+}}}

#endif // HEADER

