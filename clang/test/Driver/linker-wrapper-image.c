// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,openmp,nvptx64-nvida-cuda,sm_70
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple x86_64-unknown-linux-gnu \
// RUN:   -linker-path /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=OPENMP

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

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%S/Inputs/dummy-elf.o,cuda,nvptx64-nvida-cuda,sm_70
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple x86_64-unknown-linux-gnu \
// RUN:   -linker-path /usr/bin/ld -- %t.o -o a.out 2>&1 | FileCheck %s --check-prefix=CUDA

// CUDA: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".nv_fatbin"
// CUDA-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1180844977, i32 1, i8* getelementptr inbounds ([0 x i8], [0 x i8]* @.fatbin_image, i32 0, i32 0), i8* null }, section ".nvFatBinSegment", align 8
// CUDA-NEXT: @__dummy.cuda_offloading.entry = hidden constant [0 x %__tgt_offload_entry] zeroinitializer, section "cuda_offloading_entries"
// CUDA-NEXT: @.cuda.binary_handle = internal global i8** null
// CUDA-NEXT: @__start_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @__stop_cuda_offloading_entries = external hidden constant [0 x %__tgt_offload_entry]
// CUDA-NEXT: @llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 1, void ()* @.cuda.fatbin_reg, i8* null }]

// CUDA: define internal void @.cuda.fatbin_reg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = call i8** @__cudaRegisterFatBinary(i8* bitcast (%fatbin_wrapper* @.fatbin_wrapper to i8*))
// CUDA-NEXT:   store i8** %0, i8*** @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @.cuda.globals_reg(i8** %0)
// CUDA-NEXT:   call void @__cudaRegisterFatBinaryEnd(i8** %0)
// CUDA-NEXT:   %1 = call i32 @atexit(void ()* @.cuda.fatbin_unreg)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

// CUDA: define internal void @.cuda.fatbin_unreg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = load i8**, i8*** @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @__cudaUnregisterFatBinary(i8** %0)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

// CUDA: define internal void @.cuda.globals_reg(i8** %0) section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   br i1 icmp ne ([0 x %__tgt_offload_entry]* @__start_cuda_offloading_entries, [0 x %__tgt_offload_entry]* @__stop_cuda_offloading_entries), label %while.entry, label %while.end

// CUDA: while.entry:
// CUDA-NEXT:   %entry1 = phi %__tgt_offload_entry* [ getelementptr inbounds ([0 x %__tgt_offload_entry], [0 x %__tgt_offload_entry]* @__start_cuda_offloading_entries, i64 0, i64 0), %entry ], [ %7, %if.end ]
// CUDA-NEXT:   %1 = getelementptr inbounds %__tgt_offload_entry, %__tgt_offload_entry* %entry1, i64 0, i32 0
// CUDA-NEXT:   %addr = load i8*, i8** %1, align 8
// CUDA-NEXT:   %2 = getelementptr inbounds %__tgt_offload_entry, %__tgt_offload_entry* %entry1, i64 0, i32 1
// CUDA-NEXT:   %name = load i8*, i8** %2, align 8
// CUDA-NEXT:   %3 = getelementptr inbounds %__tgt_offload_entry, %__tgt_offload_entry* %entry1, i64 0, i32 2
// CUDA-NEXT:   %size = load i64, i64* %3, align 4
// CUDA-NEXT:   %4 = icmp eq i64 %size, 0
// CUDA-NEXT:   br i1 %4, label %if.then, label %if.else

// CUDA: if.then:
// CUDA-NEXT:   %5 = call i32 @__cudaRegisterFunction(i8** %0, i8* %addr, i8* %name, i8* %name, i32 -1, i8* null, i8* null, i8* null, i8* null, i32* null)
// CUDA-NEXT:   br label %if.end

// CUDA: if.else:
// CUDA-NEXT:   %6 = call i32 @__cudaRegisterVar(i8** %0, i8* %addr, i8* %name, i8* %name, i32 0, i64 %size, i32 0, i32 0)
// CUDA-NEXT:   br label %if.end

// CUDA: if.end:
// CUDA-NEXT:   %7 = getelementptr inbounds %__tgt_offload_entry, %__tgt_offload_entry* %entry1, i64 1
// CUDA-NEXT:   %8 = icmp eq %__tgt_offload_entry* %7, getelementptr inbounds ([0 x %__tgt_offload_entry], [0 x %__tgt_offload_entry]* @__stop_cuda_offloading_entries, i64 0, i64 0)
// CUDA-NEXT:   br i1 %8, label %while.end, label %while.entry

// CUDA: while.end:
// CUDA-NEXT:   ret void
// CUDA-NEXT: }
