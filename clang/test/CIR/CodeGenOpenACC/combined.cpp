// RUN: %clang_cc1 -fopenacc -Wno-openacc-self-if-potential-conflict -emit-cir -fclangir %s -o - | FileCheck %s

extern "C" void acc_combined(int N) {
  // CHECK: cir.func @acc_combined(%[[ARG_N:.*]]: !s32i loc{{.*}}) {
  // CHECK-NEXT: %[[ALLOCA_N:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["N", init]
  // CHECK-NEXT: cir.store %[[ARG_N]], %[[ALLOCA_N]] : !s32i, !cir.ptr<!s32i>

#pragma acc parallel loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

#pragma acc serial loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

#pragma acc kernels loop
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop default(none)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>} loc

#pragma acc serial loop default(present)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue present>} loc

#pragma acc kernels loop default(none)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } attributes {defaultAttr = #acc<defaultvalue none>} loc

#pragma acc parallel loop seq
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {seq = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop device_type(nvidia, radeon) seq
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {seq = [#acc.device_type<nvidia>, #acc.device_type<radeon>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc kernels loop seq device_type(nvidia, radeon)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {seq = [#acc.device_type<none>]} loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop auto
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop device_type(nvidia, radeon) auto
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<nvidia>, #acc.device_type<radeon>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc kernels loop auto device_type(nvidia, radeon)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {auto_ = [#acc.device_type<none>]} loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

#pragma acc parallel loop independent
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc serial loop device_type(nvidia, radeon) independent
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<nvidia>, #acc.device_type<radeon>]} loc
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
#pragma acc kernels loop independent device_type(nvidia, radeon)
  for(unsigned I = 0; I < N; ++I);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {independent = [#acc.device_type<none>]} loc
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc

  #pragma acc parallel loop collapse(1) device_type(radeon)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {collapse = [1], collapseDeviceType = [#acc.device_type<none>]}
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

  #pragma acc serial loop collapse(1) device_type(radeon) collapse (2)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.serial combined(loop) {
  // CHECK: acc.loop combined(serial) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {collapse = [1, 2], collapseDeviceType = [#acc.device_type<none>, #acc.device_type<radeon>]}
  // CHECK: acc.yield
  // CHECK-NEXT: } loc

  #pragma acc kernels loop collapse(1) device_type(radeon, nvidia) collapse (2)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.kernels combined(loop) {
  // CHECK: acc.loop combined(kernels) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {collapse = [1, 2, 2], collapseDeviceType = [#acc.device_type<none>, #acc.device_type<radeon>, #acc.device_type<nvidia>]}
  // CHECK: acc.terminator
  // CHECK-NEXT: } loc
  #pragma acc parallel loop collapse(1) device_type(radeon, nvidia) collapse(2) device_type(host) collapse(3)
  for(unsigned I = 0; I < N; ++I)
    for(unsigned J = 0; J < N; ++J)
      for(unsigned K = 0; K < N; ++K);
  // CHECK: acc.parallel combined(loop) {
  // CHECK: acc.loop combined(parallel) {
  // CHECK: acc.yield
  // CHECK-NEXT: } attributes {collapse = [1, 2, 2, 3], collapseDeviceType = [#acc.device_type<none>, #acc.device_type<radeon>, #acc.device_type<nvidia>, #acc.device_type<host>]}
  // CHECK: acc.yield
  // CHECK-NEXT: } loc
}
