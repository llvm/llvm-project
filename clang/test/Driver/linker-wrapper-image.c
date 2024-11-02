// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: clang-offload-packager -o %t.out --image=file=%S/Inputs/dummy-elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=OPENMP

//      OPENMP: @__start_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__stop_omp_offloading_entries = external hidden constant %__tgt_offload_entry
// OPENMP-NEXT: @__dummy.omp_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "omp_offloading_entries"
// OPENMP-NEXT: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"\10\FF\10\AD{{.*}}"
// OPENMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr @.omp_offloading.device_image, ptr getelementptr inbounds ([[[SIZE]] x i8], ptr @.omp_offloading.device_image, i64 1, i64 0), ptr @__start_omp_offloading_entries, ptr @__stop_omp_offloading_entries }]
// OPENMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_omp_offloading_entries, ptr @__stop_omp_offloading_entries }
// OPENMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.omp_offloading.descriptor_reg, ptr null }]
// OPENMP-NEXT: @llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.omp_offloading.descriptor_unreg, ptr null }]

//      OPENMP: define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

//      OPENMP: define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

// RUN: clang-offload-packager -o %t.out --image=file=%S/Inputs/dummy-elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

//      CUDA: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".nv_fatbin"
// CUDA-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1180844977, i32 1, ptr @.fatbin_image, ptr null }, section ".nvFatBinSegment", align 8
// CUDA-NEXT: @__dummy.cuda_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "cuda_offloading_entries"
// CUDA-NEXT: @.cuda.binary_handle = internal global ptr null
// CUDA-NEXT: @__start_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @__stop_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.cuda.fatbin_reg, ptr null }]

//      CUDA: define internal void @.cuda.fatbin_reg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = call ptr @__cudaRegisterFatBinary(ptr @.fatbin_wrapper)
// CUDA-NEXT:   store ptr %0, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @.cuda.globals_reg(ptr %0)
// CUDA-NEXT:   call void @__cudaRegisterFatBinaryEnd(ptr %0)
// CUDA-NEXT:   %1 = call i32 @atexit(ptr @.cuda.fatbin_unreg)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

//      CUDA: define internal void @.cuda.fatbin_unreg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = load ptr, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @__cudaUnregisterFatBinary(ptr %0)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

//      CUDA: define internal void @.cuda.globals_reg(ptr %0) section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   br i1 icmp ne (ptr @__start_cuda_offloading_entries, ptr @__stop_cuda_offloading_entries), label %while.entry, label %while.end

//      CUDA: while.entry:
// CUDA-NEXT:  %entry1 = phi ptr [ @__start_cuda_offloading_entries, %entry ], [ %7, %if.end ]
// CUDA-NEXT:  %1 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 0
// CUDA-NEXT:  %addr = load ptr, ptr %1, align 8
// CUDA-NEXT:  %2 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 1
// CUDA-NEXT:  %name = load ptr, ptr %2, align 8
// CUDA-NEXT:  %3 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 2
// CUDA-NEXT:  %size = load i64, ptr %3, align 4
// CUDA-NEXT:  %4 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 3
// CUDA-NEXT:  %flag = load i32, ptr %4, align 4
// CUDA-NEXT:  %5 = icmp eq i64 %size, 0
// CUDA-NEXT:  br i1 %5, label %if.then, label %if.else

//      CUDA: if.then:
// CUDA-NEXT:   %6 = call i32 @__cudaRegisterFunction(ptr %0, ptr %addr, ptr %name, ptr %name, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// CUDA-NEXT:   br label %if.end

//      CUDA: if.else:
// CUDA-NEXT:   switch i32 %flag, label %if.end [
// CUDA-NEXT:     i32 0, label %sw.global
// CUDA-NEXT:     i32 1, label %sw.managed
// CUDA-NEXT:     i32 2, label %sw.surface
// CUDA-NEXT:     i32 3, label %sw.texture
// CUDA-NEXT:   ]

//      CUDA: sw.global:
// CUDA-NEXT:   call void @__cudaRegisterVar(ptr %0, ptr %addr, ptr %name, ptr %name, i32 0, i64 %size, i32 0, i32 0)
// CUDA-NEXT:   br label %if.end

//      CUDA: sw.managed:
// CUDA-NEXT:   br label %if.end

//      CUDA: sw.surface:
// CUDA-NEXT:   br label %if.end

//      CUDA: sw.texture:
// CUDA-NEXT:   br label %if.end

//      CUDA: if.end:
// CUDA-NEXT:   %7 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 1
// CUDA-NEXT:   %8 = icmp eq ptr %7, @__stop_cuda_offloading_entries
// CUDA-NEXT:   br i1 %8, label %while.end, label %while.entry

//      CUDA: while.end:
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

// RUN: clang-offload-packager -o %t.out --image=file=%S/Inputs/dummy-elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=HIP

//      HIP: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".hip_fatbin"
// HIP-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1212764230, i32 1, ptr @.fatbin_image, ptr null }, section ".hipFatBinSegment", align 8
// HIP-NEXT: @__dummy.hip_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "hip_offloading_entries"
// HIP-NEXT: @.hip.binary_handle = internal global ptr null
// HIP-NEXT: @__start_hip_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// HIP-NEXT: @__stop_hip_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// HIP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 1, ptr @.hip.fatbin_reg, ptr null }]

//      HIP: define internal void @.hip.fatbin_reg() section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   %0 = call ptr @__hipRegisterFatBinary(ptr @.fatbin_wrapper)
// HIP-NEXT:   store ptr %0, ptr @.hip.binary_handle, align 8
// HIP-NEXT:   call void @.hip.globals_reg(ptr %0)
// HIP-NEXT:   %1 = call i32 @atexit(ptr @.hip.fatbin_unreg)
// HIP-NEXT:   ret void
// HIP-NEXT: }

//      HIP: define internal void @.hip.fatbin_unreg() section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   %0 = load ptr, ptr @.hip.binary_handle, align 8
// HIP-NEXT:   call void @__hipUnregisterFatBinary(ptr %0)
// HIP-NEXT:   ret void
// HIP-NEXT: }

//      HIP: define internal void @.hip.globals_reg(ptr %0) section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   br i1 icmp ne (ptr @__start_hip_offloading_entries, ptr @__stop_hip_offloading_entries), label %while.entry, label %while.end

//      HIP: while.entry:
// HIP-NEXT:   %entry1 = phi ptr [ @__start_hip_offloading_entries, %entry ], [ %7, %if.end ]
// HIP-NEXT:   %1 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 0
// HIP-NEXT:   %addr = load ptr, ptr %1, align 8
// HIP-NEXT:   %2 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 1
// HIP-NEXT:   %name = load ptr, ptr %2, align 8
// HIP-NEXT:   %3 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 2
// HIP-NEXT:   %size = load i64, ptr %3, align 4
// HIP-NEXT:   %4 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 0, i32 3
// HIP-NEXT:   %flag = load i32, ptr %4, align 4
// HIP-NEXT:   %5 = icmp eq i64 %size, 0
// HIP-NEXT:   br i1 %5, label %if.then, label %if.else

//      HIP: if.then:
// HIP-NEXT:   %6 = call i32 @__hipRegisterFunction(ptr %0, ptr %addr, ptr %name, ptr %name, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// HIP-NEXT:   br label %if.end

//      HIP: if.else:
// HIP-NEXT:   switch i32 %flag, label %if.end [
// HIP-NEXT:     i32 0, label %sw.global
// HIP-NEXT:     i32 1, label %sw.managed
// HIP-NEXT:     i32 2, label %sw.surface
// HIP-NEXT:     i32 3, label %sw.texture
// HIP-NEXT:   ]

//      HIP: sw.global:
// HIP-NEXT:   call void @__hipRegisterVar(ptr %0, ptr %addr, ptr %name, ptr %name, i32 0, i64 %size, i32 0, i32 0)
// HIP-NEXT:   br label %if.end

//      HIP: sw.managed:
// HIP-NEXT:   br label %if.end

//      HIP: sw.surface:
// HIP-NEXT:   br label %if.end

//      HIP: sw.texture:
// HIP-NEXT:   br label %if.end

//      HIP: if.end:
// HIP-NEXT:   %7 = getelementptr inbounds %__tgt_offload_entry, ptr %entry1, i64 1
// HIP-NEXT:   %8 = icmp eq ptr %7, @__stop_hip_offloading_entries
// HIP-NEXT:   br i1 %8, label %while.end, label %while.entry

//      HIP: while.end:
// HIP-NEXT:   ret void
// HIP-NEXT: }
