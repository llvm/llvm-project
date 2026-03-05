#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM-HOST --input-file=%t.ll %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:            -fcuda-is-device -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-DEVICE --input-file=%t.ll %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu \
// RUN:            -x cuda -emit-llvm -target-sdk-version=12.3 \
// RUN:            %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG-HOST --input-file=%t.ll %s

// Make sure we do emit device-side kernel even if it's only referenced
// by the destructor of a variable not present on device.
template<typename T> __global__ void f(T) {}
template<typename T> struct A {
  ~A() { f<<<1, 1>>>(T()); }
};

// CIR-HOST: module
// CIR-DEVICE: module
// CIR-DEVICE: cir.func {{.*}} @_Z1fIiEvT_
// LLVM-DEVICE: define dso_local ptx_kernel void @_Z1fIiEvT_
// OGCG-DEVICE: define ptx_kernel void @_Z1fIiEvT_

// CIR-HOST: cir.func {{.*}} @_ZN1AIiED2Ev{{.*}} {
// CIR-HOST:   cir.call @__cudaPushCallConfiguration
// CIR-HOST:   cir.call @_Z16__device_stub__fIiEvT_
// CIR-HOST: }

// LLVM-HOST: define linkonce_odr void @_ZN1AIiED2Ev
// LLVM-HOST: call i32 @__cudaPushCallConfiguration(
// LLVM-HOST: call void @_Z16__device_stub__fIiEvT_

// OGCG-HOST: define linkonce_odr void @_ZN1AIiED2Ev
// OGCG-HOST: call i32 @__cudaPushCallConfiguration(
// OGCG-HOST: call void @_Z16__device_stub__fIiEvT_



A<int> a;
