// RUN: %clang_cc1 -verify -fopenmp -fblocks -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fblocks -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=SIMD-ONLY
// RUN: %clang_cc1 -verify -fopenmp -fblocks -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fblocks -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fblocks -triple x86_64-unknown-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -o - | FileCheck %s --check-prefix=DEVICE
// expected-no-diagnostics

// Target regions inside lambdas and blocks at namespace scope have no parent
// function declaration, so they are named after the function being emitted.
// The device compilation finds such lambdas through the declarations holding them.

int x;
int &ref = x;

auto lambda = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = 42;
};

auto nested_lambda = []() {
  return []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
    ref = 42;
  };
};

auto combined_directive = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target teams
  ref = 42;
};

auto nested_in_parallel = []() {
#pragma omp parallel
  {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
    ref = 42;
  }
};

auto static_local = []() {
  static int &local_ref = x;
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  local_ref = 42;
};

// The body of a block is not scanned for the device.
void (^block)() = ^{
// CHECK-DAG: define internal void @__omp_offloading_{{.*}}_block_block_invoke_l[[#@LINE+1]](
#pragma omp target
  ref = 42;
};

void default_argument(int = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = 42;
  return 0;
}());

template <int N> int variable_template = []() {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @{{"?}}__omp_offloading_{{.*}}clEv_l[[#@LINE+1]]{{"?}}(
#pragma omp target
  ref = N;
  return 0;
}();
int instantiation = variable_template<1>;

// Reproducer from GH136652.
auto slice_work = [](__SIZE_TYPE__ device, float start, float end) {
// CHECK-DAG: define internal void @{{"?}}__omp_offloading_{{.*}}clEmff_l[[#@LINE+2]]{{"?}}(
// DEVICE-DAG: define weak_odr protected void @__omp_offloading_{{.*}}_ZNKUlmffE_clEmff_l[[#@LINE+1]](
#pragma omp target teams device(device)
  {
  }
};

struct GlobalObject {
  int member = [](int d) {
// CHECK-DAG: define internal void @__omp_offloading_{{.*}}_ZNK12GlobalObject6memberMUliE_clEi_l[[#@LINE+2]](
// DEVICE-DAG: define weak_odr protected void @__omp_offloading_{{.*}}_ZNK12GlobalObject6memberMUliE_clEi_l[[#@LINE+1]](
#pragma omp target teams device(d)
    {
    }
    return d;
  }(0);
};
GlobalObject global_object;

struct LocalObject {
  int member = [](int d) {
// CHECK-DAG: define internal void @__omp_offloading_{{.*}}_ZNK11LocalObject6memberMUliE_clEi_l[[#@LINE+2]](
// DEVICE-DAG: define weak_odr protected void @__omp_offloading_{{.*}}_ZNK11LocalObject6memberMUliE_clEi_l[[#@LINE+1]](
#pragma omp target teams device(d)
    {
    }
    return d;
  }(0);
};

template <class T> struct Holder {
  static inline auto member = [](T d) {
// CHECK-DAG: define internal void @__omp_offloading_{{.*}}_ZNK6HolderIiE6memberMUliE_clEi_l[[#@LINE+2]](
// DEVICE-DAG: define weak_odr protected void @__omp_offloading_{{.*}}_ZNK6HolderIiE6memberMUliE_clEi_l[[#@LINE+1]](
#pragma omp target teams device(d)
    {
    }
  };
};

void use() {
  lambda();
  nested_lambda()();
  combined_directive();
  nested_in_parallel();
  static_local();
  default_argument();
  slice_work(0, 0, 0);
  LocalObject local_object;
  (void)local_object;
  Holder<int>::member(0);
  // Control: named after the enclosing function.
  static auto static_local_lambda = [](__SIZE_TYPE__ device) {
// CHECK-DAG: define internal void @__omp_offloading_{{.*}}_Z3usev_l[[#@LINE+2]](
// DEVICE-DAG: define weak_odr protected void @__omp_offloading_{{.*}}_Z3usev_l[[#@LINE+1]](
#pragma omp target teams device(device)
    {
    }
  };
  static_local_lambda(0);
}

// SIMD-ONLY-NOT: {{__kmpc|__tgt}}
