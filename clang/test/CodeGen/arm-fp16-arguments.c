// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs -mfloat-abi soft -fallow-half-arguments-and-returns -emit-llvm -o - -O2 %s | FileCheck %s --check-prefix=CHECK --check-prefix=SOFT
// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs -mfloat-abi hard -fallow-half-arguments-and-returns -emit-llvm -o - -O2 %s | FileCheck %s --check-prefix=CHECK --check-prefix=HARD
// RUN: %clang_cc1 -triple armv7a--none-eabi -target-abi aapcs -mfloat-abi soft -fnative-half-arguments-and-returns -emit-llvm -o - -O2 %s | FileCheck %s --check-prefix=CHECK --check-prefix=NATIVE

__fp16 g;

void t1(__fp16 a) { g = a; }
// SOFT: define{{.*}} void @t1(half noundef [[PARAM:%.*]])
// HARD: define{{.*}} arm_aapcs_vfpcc void @t1(half noundef [[PARAM:%.*]])
// NATIVE: define{{.*}} void @t1(half noundef [[PARAM:%.*]])
// CHECK: store half [[PARAM]], half* @g

__fp16 t2() { return g; }
// SOFT: define{{.*}} half @t2()
// HARD: define{{.*}} arm_aapcs_vfpcc half @t2()
// NATIVE: define{{.*}} half @t2()
// CHECK: [[LOAD:%.*]] = load half, half* @g
// CHECK: ret half [[LOAD]]

_Float16 h;

void t3(_Float16 a) { h = a; }
// SOFT: define{{.*}} void @t3(half noundef [[PARAM:%.*]])
// HARD: define{{.*}} arm_aapcs_vfpcc void @t3(half noundef [[PARAM:%.*]])
// NATIVE: define{{.*}} void @t3(half noundef [[PARAM:%.*]])
// CHECK: store half [[PARAM]], half* @h

_Float16 t4() { return h; }
// SOFT: define{{.*}} half @t4()
// HARD: define{{.*}} arm_aapcs_vfpcc half @t4()
// NATIVE: define{{.*}} half @t4()
// CHECK: [[LOAD:%.*]] = load half, half* @h
// CHECK: ret half [[LOAD]]
