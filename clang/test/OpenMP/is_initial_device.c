// REQUIRES: powerpc-registered-target

// RUN: %clang_cc1 -verify -fopenmp -x c -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown \
// RUN:            -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x ir -triple powerpc64le-unknown-unknown -emit-llvm \
// RUN:             %t-ppc-host.bc -o - | FileCheck %s -check-prefixes HOST,OUTLINED
// RUN: %clang_cc1 -verify -fopenmp -x c -triple powerpc64le-unknown-unknown -emit-llvm -fopenmp-is-device \
// RUN:             %s -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s -check-prefixes DEVICE,OUTLINED

// expected-no-diagnostics
int check() {
  int host = omp_is_initial_device();
  int device;
#pragma omp target map(tofrom: device)
  {
    device = omp_is_initial_device();
  }

  return host + device;
}

// The host should get a value of 1:
// HOST: define{{.*}} @check()
// HOST: [[HOST:%.*]] = alloca i32
// HOST: store i32 1, i32* [[HOST]]

// OUTLINED: define{{.*}} @{{.*}}omp_offloading{{.*}}(i32*{{.*}} [[DEVICE_ARGUMENT:%.*]])
// OUTLINED: [[DEVICE_ADDR_STORAGE:%.*]] = alloca i32*
// OUTLINED: store i32* [[DEVICE_ARGUMENT]], i32** [[DEVICE_ADDR_STORAGE]]
// OUTLINED: [[DEVICE_ADDR:%.*]] = load i32*, i32** [[DEVICE_ADDR_STORAGE]]

// The outlined function that is called as fallback also runs on the host:
// HOST: store i32 1, i32* [[DEVICE_ADDR]]

// The device should get a value of 0:
// DEVICE: store i32 0, i32* [[DEVICE_ADDR]]
