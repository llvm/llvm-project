// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -cc1 %s -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,nvptx64-nvida-cuda,sm_70
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run -linker-path /usr/bin/ld \
// RUN:   -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=OPENMP

// OPENMP: @__start_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__stop_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__dummy.omp_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "omp_offloading_entries"
// OPENMP-NEXT: @.omp_offloading.device_image = internal unnamed_addr constant [0 x i8] zeroinitializer
// OPENMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { i8* getelementptr inbounds ([0 x i8], [0 x i8]* @.omp_offloading.device_image, i64 0, i64 0), i8* getelementptr inbounds ([0 x i8], [0 x i8]* @.omp_offloading.device_image, i64 0, i64 0), %__tgt_offload_entry* @__start_omp_offloading_entries, %__tgt_offload_entry* @__stop_omp_offloading_entries }]
// OPENMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, %__tgt_device_image* getelementptr inbounds ([1 x %__tgt_device_image], [1 x %__tgt_device_image]* @.omp_offloading.device_images, i64 0, i64 0), %__tgt_offload_entry* @__start_omp_offloading_entries, %__tgt_offload_entry* @__stop_omp_offloading_entries }
// OPENMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @.omp_offloading.descriptor_reg, i8* null }]
// OPENMP-NEXT: @llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @.omp_offloading.descriptor_unreg, i8* null }]

// OPENMP: define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_register_lib(%__tgt_bin_desc* @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

// OPENMP: define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_unregister_lib(%__tgt_bin_desc* @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }
