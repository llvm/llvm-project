// Test OpenMP module attributes in CIR output.

// Host, x86_64, no target triples
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST

// HOST: module {{.*}} attributes {
// HOST-SAME: omp.is_gpu = false
// HOST-SAME: omp.is_target_device = false

// Host with target triples
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda %s -o - \
// RUN:   | FileCheck %s --check-prefix=HOST-TRIPLES

// HOST-TRIPLES: module {{.*}} attributes {
// HOST-TRIPLES-SAME: omp.is_gpu = false
// HOST-TRIPLES-SAME: omp.is_target_device = false
// HOST-TRIPLES-SAME: omp.target_triples = ["amdgcn-amd-amdhsa", "nvptx64-nvidia-cuda"]

// Device, AMDGPU
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fclangir -emit-cir \
// RUN:   -fopenmp-is-target-device %s -o - \
// RUN:   | FileCheck %s --check-prefix=AMDGPU-DEVICE

// AMDGPU-DEVICE: module {{.*}} attributes {
// AMDGPU-DEVICE-SAME: omp.is_gpu = true
// AMDGPU-DEVICE-SAME: omp.is_target_device = true

// Device, NVPTX
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fopenmp -fclangir -emit-cir \
// RUN:   -fopenmp-is-target-device %s -o - \
// RUN:   | FileCheck %s --check-prefix=NVPTX-DEVICE

// NVPTX-DEVICE: module {{.*}} attributes {
// NVPTX-DEVICE-SAME: omp.is_gpu = true
// NVPTX-DEVICE-SAME: omp.is_target_device = true

// Device, CPU
// RUN: %clang_cc1 -fopenmp -fclangir -emit-cir \
// RUN:   -fopenmp-is-target-device %s -o - \
// RUN:   | FileCheck %s --check-prefix=CPU-DEVICE

// CPU-DEVICE: module {{.*}} attributes {
// CPU-DEVICE-SAME: omp.is_gpu = false
// CPU-DEVICE-SAME: omp.is_target_device = true

// Device with omp.flags
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fopenmp -fclangir -emit-cir \
// RUN:   -fopenmp-is-target-device \
// RUN:   -fopenmp-assume-no-thread-state \
// RUN:   -fopenmp-assume-no-nested-parallelism %s -o - \
// RUN:   | FileCheck %s --check-prefix=DEVICE-FLAGS

// DEVICE-FLAGS: module {{.*}}  attributes {
// DEVICE-FLAGS-SAME: omp.flags = #omp.flags<assume_no_thread_state = true, assume_no_nested_parallelism = true
// DEVICE-FLAGS-SAME: omp.is_gpu = true
// DEVICE-FLAGS-SAME: omp.is_target_device = true

// Force USM (host)
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fopenmp -fclangir -emit-cir \
// RUN:   -fopenmp-force-usm %s -o - \
// RUN:   | FileCheck %s --check-prefix=USM

// USM: module {{.*}}  attributes {
// USM-SAME: omp.requires = #omp<clause_requires unified_shared_memory>

void omp_function(void) {}
