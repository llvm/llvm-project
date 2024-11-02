// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: clang-offload-packager -o %t.out --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple x86_64-unknown-linux-gnu \
// RUN:   -linker-path /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=OPENMP

// OPENMP: @__start_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__stop_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__dummy.omp_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "omp_offloading_entries"
// OPENMP-NEXT: @.omp_offloading.device_image = internal unnamed_addr constant [0 x i8] zeroinitializer
// OPENMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr @.omp_offloading.device_image, ptr @.omp_offloading.device_image, ptr @__start_omp_offloading_entries, ptr @__stop_omp_offloading_entries }]
// OPENMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_omp_offloading_entries, ptr @__stop_omp_offloading_entries }
// OPENMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.omp_offloading.descriptor_reg, ptr null }]
// OPENMP-NEXT: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.omp_offloading.descriptor_unreg, ptr null }]

// OPENMP: define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

// OPENMP: define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

// RUN: clang-offload-packager -o %t.out --image=file=%S/Inputs/dummy-elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple x86_64-unknown-linux-gnu \
// RUN:   -linker-path /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

// CUDA: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".nv_fatbin"
// CUDA-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1180844977, i32 1, ptr @.fatbin_image, ptr null }, section ".nvFatBinSegment", align 8
// CUDA-NEXT: @__dummy.cuda_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "cuda_offloading_entries"
// CUDA-NEXT: @.cuda.binary_handle = internal global ptr null
// CUDA-NEXT: @__start_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @__stop_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.cuda.fatbin_reg, ptr null }]

// CUDA: define internal void @.cuda.fatbin_reg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = call ptr @__cudaRegisterFatBinary(ptr @.fatbin_wrapper)
// CUDA-NEXT:   store ptr %0, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @.cuda.globals_reg(ptr %0)
// CUDA-NEXT:   call void @__cudaRegisterFatBinaryEnd(ptr %0)
// CUDA-NEXT:   %1 = call i32 @atexit(ptr @.cuda.fatbin_unreg)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

// CUDA: define internal void @.cuda.fatbin_unreg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = load ptr, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @__cudaUnregisterFatBinary(ptr %0)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

// CUDA: define internal void @.cuda.globals_reg(ptr %0) section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   br i1 icmp ne (ptr @__start_cuda_offloading_entries, ptr @__stop_cuda_offloading_entries), label %while.entry, label %while.end

// CUDA: while.entry:
// CUDA-NEXT:   %entry1 = phi ptr [ @__start_cuda_offloading_entries, %entry ], [ %7, %if.end ]
// CUDA-NEXT:   %1 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 0
// CUDA-NEXT:   %addr = load ptr, ptr %1, align 8
// CUDA-NEXT:   %2 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 1
// CUDA-NEXT:   %name = load ptr, ptr %2, align 8
// CUDA-NEXT:   %3 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 2
// CUDA-NEXT:   %size = load i64, ptr %3, align 4
// CUDA-NEXT:   %4 = icmp eq i64 %size, 0
// CUDA-NEXT:   br i1 %4, label %if.then, label %if.else

// CUDA: if.then:
// CUDA-NEXT:   %5 = call i32 @__cudaRegisterFunction(ptr %0, ptr %addr, ptr %name, ptr %name, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// CUDA-NEXT:   br label %if.end

// CUDA: if.else:
// CUDA-NEXT:   %6 = call i32 @__cudaRegisterVar(ptr %0, ptr %addr, ptr %name, ptr %name, i32 0, i64 %size, i32 0, i32 0)
// CUDA-NEXT:   br label %if.end

// CUDA: if.end:
// CUDA-NEXT:   %7 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 1
// CUDA-NEXT:   %8 = icmp eq ptr %7, @__stop_cuda_offloading_entries
// CUDA-NEXT:   br i1 %8, label %while.end, label %while.entry

// CUDA: while.end:
// CUDA-NEXT:   ret void
// CUDA-NEXT: }
