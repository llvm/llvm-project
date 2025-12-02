// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
// REQUIRES: amdgpu-registered-target
// REQUIRES: spirv-registered-target

// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.elf.o

// RUN: llvm-offload-binary -o %t.out --image=file=%t.elf.o,kind=openmp,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=OPENMP,OPENMP-ELF
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run -r --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=OPENMP-ELF,OPENMP-REL
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-windows-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=OPENMP,OPENMP-COFF

//      OPENMP-ELF: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
// OPENMP-ELF-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
// OPENMP-ELF-NEXT: @__dummy.llvm_offload_entries = internal constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries"

//      OPENMP-COFF: @__start_llvm_offload_entries = weak_odr hidden constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OA"
// OPENMP-COFF-NEXT: @__stop_llvm_offload_entries = weak_odr hidden constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OZ"

// OPENMP-REL: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"\10\FF\10\AD{{.*}}", section ".llvm.offloading.relocatable", align 8

//      OPENMP: @.omp_offloading.device_image = internal unnamed_addr constant [[[SIZE:[0-9]+]] x i8] c"\10\FF\10\AD{{.*}}", section ".llvm.offloading", align 8
// OPENMP-NEXT: @.omp_offloading.device_images = internal unnamed_addr constant [1 x %__tgt_device_image] [%__tgt_device_image { ptr getelementptr ([[[BEGIN:[0-9]+]] x i8], ptr @.omp_offloading.device_image, i64 0, i64 144), ptr getelementptr ([[[END:[0-9]+]] x i8], ptr @.omp_offloading.device_image, i64 0, i64 144), ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }]
// OPENMP-NEXT: @.omp_offloading.descriptor = internal constant %__tgt_bin_desc { i32 1, ptr @.omp_offloading.device_images, ptr @__start_llvm_offload_entries, ptr @__stop_llvm_offload_entries }
// OPENMP-NEXT: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.omp_offloading.descriptor_reg, ptr null }]

//      OPENMP: define internal void @.omp_offloading.descriptor_reg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_register_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   %0 = call i32 @atexit(ptr @.omp_offloading.descriptor_unreg)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

//      OPENMP: define internal void @.omp_offloading.descriptor_unreg() section ".text.startup" {
// OPENMP-NEXT: entry:
// OPENMP-NEXT:   call void @__tgt_unregister_lib(ptr @.omp_offloading.descriptor)
// OPENMP-NEXT:   ret void
// OPENMP-NEXT: }

// RUN: llvm-offload-binary -o %t.out --image=file=%t.elf.o,kind=cuda,triple=nvptx64-nvidia-cuda,arch=sm_70
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=CUDA,CUDA-ELF
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run -r --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=CUDA,CUDA-ELF
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-windows-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=CUDA,CUDA-COFF

//      CUDA-ELF: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
// CUDA-ELF-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
// CUDA-ELF-NEXT: @__dummy.llvm_offload_entries = internal constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries"

//      CUDA-COFF: @__start_llvm_offload_entries = weak_odr hidden constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OA"
// CUDA-COFF-NEXT: @__stop_llvm_offload_entries = weak_odr hidden constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OZ"

//      CUDA: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".nv_fatbin"
// CUDA-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1180844977, i32 1, ptr @.fatbin_image, ptr null }, section ".nvFatBinSegment", align 8
// CUDA-NEXT: @.cuda.binary_handle = internal global ptr null

// CUDA: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.cuda.fatbin_reg, ptr null }]

//      CUDA: define internal void @.cuda.fatbin_reg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = call ptr @__cudaRegisterFatBinary(ptr @.fatbin_wrapper)
// CUDA-NEXT:   store ptr %0, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @.cuda.globals_reg(ptr %0)
// CUDA-NEXT:   call void @__cudaRegisterFatBinaryEnd(ptr %0)
// CUDA-NEXT:   %1 = call i32 @atexit(ptr @.cuda.fatbin_unreg)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }
//
//      CUDA: define internal void @.cuda.fatbin_unreg() section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %0 = load ptr, ptr @.cuda.binary_handle, align 8
// CUDA-NEXT:   call void @__cudaUnregisterFatBinary(ptr %0)
// CUDA-NEXT:   ret void
// CUDA-NEXT: }
//
//      CUDA: define internal void @.cuda.globals_reg(ptr %0) section ".text.startup" {
// CUDA-NEXT: entry:
// CUDA-NEXT:   %1 = icmp ne ptr @__start_llvm_offload_entries, @__stop_llvm_offload_entries
// CUDA-NEXT:   br i1 %1, label %while.entry, label %while.end
//
//      CUDA: while.entry:
// CUDA-NEXT:   %entry1 = phi ptr [ @__start_llvm_offload_entries, %entry ], [ %16, %if.end ]
// CUDA-NEXT:   %2 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 4
// CUDA-NEXT:   %addr = load ptr, ptr %2, align 8
// CUDA-NEXT:   %3 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 8
// CUDA-NEXT:   %aux_addr = load ptr, ptr %3, align 8
// CUDA-NEXT:   %4 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 2
// CUDA-NEXT:   %kind = load i16, ptr %4, align 2
// CUDA-NEXT:   %5 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 5
// CUDA-NEXT:   %name = load ptr, ptr %5, align 8
// CUDA-NEXT:   %6 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 6
// CUDA-NEXT:   %size = load i64, ptr %6, align 4
// CUDA-NEXT:   %7 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 3
// CUDA-NEXT:   %flags = load i32, ptr %7, align 4
// CUDA-NEXT:   %8 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 7
// CUDA-NEXT:   %data = load i64, ptr %8, align 4
// CUDA-NEXT:   %9 = trunc i64 %data to i32
// CUDA-NEXT:   %type = and i32 %flags, 7
// CUDA-NEXT:   %10 = and i32 %flags, 8
// CUDA-NEXT:   %extern = lshr i32 %10, 3
// CUDA-NEXT:   %11 = and i32 %flags, 16
// CUDA-NEXT:   %constant = lshr i32 %11, 4
// CUDA-NEXT:   %12 = and i32 %flags, 32
// CUDA-NEXT:   %normalized = lshr i32 %12, 5
// CUDA-NEXT:   %13 = icmp eq i16 %kind, 2
// CUDA-NEXT:   br i1 %13, label %if.kind, label %if.end
//
//      CUDA: if.kind:
// CUDA-NEXT:   %14 = icmp eq i64 %size, 0
// CUDA-NEXT:   br i1 %14, label %if.then, label %if.else
//
//      CUDA: if.then:
// CUDA-NEXT:   %15 = call i32 @__cudaRegisterFunction(ptr %0, ptr %addr, ptr %name, ptr %name, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// CUDA-NEXT:   br label %if.end
//
//      CUDA: if.else:
// CUDA-NEXT:   switch i32 %type, label %if.end [
// CUDA-NEXT:     i32 0, label %sw.global
// CUDA-NEXT:     i32 1, label %sw.managed
// CUDA-NEXT:     i32 2, label %sw.surface
// CUDA-NEXT:     i32 3, label %sw.texture
// CUDA-NEXT:   ]
//
//      CUDA: sw.global:
// CUDA-NEXT:   call void @__cudaRegisterVar(ptr %0, ptr %addr, ptr %name, ptr %name, i32 %extern, i64 %size, i32 %constant, i32 0)
// CUDA-NEXT:   br label %if.end
//
//      CUDA: sw.managed:
// CUDA-NEXT:   call void @__cudaRegisterManagedVar(ptr %0, ptr %aux_addr, ptr %addr, ptr %name, i64 %size, i32 %9)
// CUDA-NEXT:   br label %if.end
//
//      CUDA: sw.surface:
// CUDA-NEXT:   br label %if.end
//
//      CUDA: sw.texture:
// CUDA-NEXT:   br label %if.end
//
//      CUDA: if.end:
// CUDA-NEXT:   %16 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i64 1
// CUDA-NEXT:   %17 = icmp eq ptr %16, @__stop_llvm_offload_entries
// CUDA-NEXT:   br i1 %17, label %while.end, label %while.entry
//
//      CUDA: while.end:
// CUDA-NEXT:   ret void
// CUDA-NEXT: }

// RUN: llvm-offload-binary -o %t.out --image=file=%t.elf.o,kind=hip,triple=amdgcn-amd-amdhsa,arch=gfx908
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=HIP,HIP-ELF
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu -r \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=HIP,HIP-ELF
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-windows-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=HIP,HIP-COFF

//      HIP-ELF: @__start_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
// HIP-ELF-NEXT: @__stop_llvm_offload_entries = external hidden constant [0 x %struct.__tgt_offload_entry]
// HIP-ELF-NEXT: @__dummy.llvm_offload_entries = internal constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries"

//      HIP-COFF: @__start_llvm_offload_entries = weak_odr hidden constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OA"
// HIP-COFF-NEXT: @__stop_llvm_offload_entries = weak_odr hidden constant [0 x %struct.__tgt_offload_entry] zeroinitializer, section "llvm_offload_entries$OZ"

//      HIP: @.fatbin_image = internal constant [0 x i8] zeroinitializer, section ".hip_fatbin"
// HIP-NEXT: @.fatbin_wrapper = internal constant %fatbin_wrapper { i32 1212764230, i32 1, ptr @.fatbin_image, ptr null }, section ".hipFatBinSegment", align 8
// HIP-NEXT: @.hip.binary_handle = internal global ptr null

// HIP: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 101, ptr @.hip.fatbin_reg, ptr null }]

//      HIP: define internal void @.hip.fatbin_reg() section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   %0 = call ptr @__hipRegisterFatBinary(ptr @.fatbin_wrapper)
// HIP-NEXT:   store ptr %0, ptr @.hip.binary_handle, align 8
// HIP-NEXT:   call void @.hip.globals_reg(ptr %0)
// HIP-NEXT:   %1 = call i32 @atexit(ptr @.hip.fatbin_unreg)
// HIP-NEXT:   ret void
// HIP-NEXT: }
//
//      HIP: define internal void @.hip.fatbin_unreg() section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   %0 = load ptr, ptr @.hip.binary_handle, align 8
// HIP-NEXT:   call void @__hipUnregisterFatBinary(ptr %0)
// HIP-NEXT:   ret void
// HIP-NEXT: }
//
//      HIP: define internal void @.hip.globals_reg(ptr %0) section ".text.startup" {
// HIP-NEXT: entry:
// HIP-NEXT:   %1 = icmp ne ptr @__start_llvm_offload_entries, @__stop_llvm_offload_entries
// HIP-NEXT:   br i1 %1, label %while.entry, label %while.end
//
//      HIP: while.entry:
// HIP-NEXT:   %entry1 = phi ptr [ @__start_llvm_offload_entries, %entry ], [ %16, %if.end ]
// HIP-NEXT:   %2 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 4
// HIP-NEXT:   %addr = load ptr, ptr %2, align 8
// HIP-NEXT:   %3 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 8
// HIP-NEXT:   %aux_addr = load ptr, ptr %3, align 8
// HIP-NEXT:   %4 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 2
// HIP-NEXT:   %kind = load i16, ptr %4, align 2
// HIP-NEXT:   %5 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 5
// HIP-NEXT:   %name = load ptr, ptr %5, align 8
// HIP-NEXT:   %6 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 6
// HIP-NEXT:   %size = load i64, ptr %6, align 4
// HIP-NEXT:   %7 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 3
// HIP-NEXT:   %flags = load i32, ptr %7, align 4
// HIP-NEXT:   %8 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i32 0, i32 7
// HIP-NEXT:   %data = load i64, ptr %8, align 4
// HIP-NEXT:   %9 = trunc i64 %data to i32
// HIP-NEXT:   %type = and i32 %flags, 7
// HIP-NEXT:   %10 = and i32 %flags, 8
// HIP-NEXT:   %extern = lshr i32 %10, 3
// HIP-NEXT:   %11 = and i32 %flags, 16
// HIP-NEXT:   %constant = lshr i32 %11, 4
// HIP-NEXT:   %12 = and i32 %flags, 32
// HIP-NEXT:   %normalized = lshr i32 %12, 5
// HIP-NEXT:   %13 = icmp eq i16 %kind, 4
// HIP-NEXT:   br i1 %13, label %if.kind, label %if.end
//
//      HIP: if.kind:
// HIP-NEXT:   %14 = icmp eq i64 %size, 0
// HIP-NEXT:   br i1 %14, label %if.then, label %if.else
//
//      HIP: if.then:
// HIP-NEXT:   %15 = call i32 @__hipRegisterFunction(ptr %0, ptr %addr, ptr %name, ptr %name, i32 -1, ptr null, ptr null, ptr null, ptr null, ptr null)
// HIP-NEXT:   br label %if.end
//
//      HIP: if.else:
// HIP-NEXT:   switch i32 %type, label %if.end [
// HIP-NEXT:     i32 0, label %sw.global
// HIP-NEXT:     i32 1, label %sw.managed
// HIP-NEXT:     i32 2, label %sw.surface
// HIP-NEXT:     i32 3, label %sw.texture
// HIP-NEXT:   ]
//
//      HIP: sw.global:
// HIP-NEXT:   call void @__hipRegisterVar(ptr %0, ptr %addr, ptr %name, ptr %name, i32 %extern, i64 %size, i32 %constant, i32 0)
// HIP-NEXT:   br label %if.end
//
//      HIP: sw.managed:
// HIP-NEXT:   call void @__hipRegisterManagedVar(ptr %0, ptr %aux_addr, ptr %addr, ptr %name, i64 %size, i32 %9)
// HIP-NEXT:   br label %if.end
//
//      HIP: sw.surface:
// HIP-NEXT:   call void @__hipRegisterSurface(ptr %0, ptr %addr, ptr %name, ptr %name, i32 %9, i32 %extern)
// HIP-NEXT:   br label %if.end
//
//      HIP: sw.texture:
// HIP-NEXT:   call void @__hipRegisterTexture(ptr %0, ptr %addr, ptr %name, ptr %name, i32 %9, i32 %normalized, i32 %extern)
// HIP-NEXT:   br label %if.end
//
//      HIP: if.end:
// HIP-NEXT:   %16 = getelementptr inbounds %struct.__tgt_offload_entry, ptr %entry1, i64 1
// HIP-NEXT:   %17 = icmp eq ptr %16, @__stop_llvm_offload_entries
// HIP-NEXT:   br i1 %17, label %while.end, label %while.entry
//
//      HIP: while.end:
// HIP-NEXT:   ret void
// HIP-NEXT: }

// RUN: llvm-offload-binary -o %t.out --image=file=%t.elf.o,kind=sycl,triple=spirv64-unknown-unknown,arch=generic
// RUN: %clang -cc1 %s -triple x86_64-unknown-linux-gnu -emit-obj -o %t.o \
// RUN:   -fembed-offload-object=%t.out
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=SYCL
// RUN: clang-linker-wrapper --print-wrapped-module --dry-run --host-triple=x86_64-unknown-linux-gnu -r \
// RUN:   --linker-path=/usr/bin/ld %t.o -o a.out 2>&1 | FileCheck %s --check-prefixes=SYCL

//      SYCL: %__sycl.tgt_device_image = type { i16, i8, i8, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
// SYCL-NEXT: %__sycl.tgt_bin_desc = type { i16, i16, ptr, ptr, ptr }

//      SYCL: @.sycl_offloading.target.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// SYCL-NEXT: @.sycl_offloading.opts.compile.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// SYCL-NEXT: @.sycl_offloading.opts.link.0 = internal unnamed_addr constant [1 x i8] zeroinitializer
// SYCL-NEXT: @.sycl_offloading.0.data = internal unnamed_addr constant [0 x i8] zeroinitializer, section ".llvm.offloading"
// SYCL-NEXT: @.offloading.entry_name = internal unnamed_addr constant [5 x i8] c"stub\00", section ".llvm.rodata.offloading", align 1
// SYCL-NEXT: @.offloading.entry.stub = weak constant %struct.__tgt_offload_entry { i64 0, i16 1, i16 8, i32 0, ptr null, ptr @.offloading.entry_name, i64 0, i64 0, ptr null }, section "llvm_offload_entries", align 8
// SYCL-NEXT: @.sycl_offloading.entries_arr = internal constant [1 x %struct.__tgt_offload_entry] [%struct.__tgt_offload_entry { i64 0, i16 1, i16 8, i32 0, ptr null, ptr @.offloading.entry_name, i64 0, i64 0, ptr null }]
// SYCL-NEXT: @.sycl_offloading.device_images = internal unnamed_addr constant [1 x %__sycl.tgt_device_image] [%__sycl.tgt_device_image { i16 3, i8 8, i8 0, ptr @.sycl_offloading.target.0, ptr @.sycl_offloading.opts.compile.0, ptr @.sycl_offloading.opts.link.0, ptr @.sycl_offloading.0.data, ptr @.sycl_offloading.0.data, ptr @.sycl_offloading.entries_arr, ptr getelementptr ([1 x %struct.__tgt_offload_entry], ptr @.sycl_offloading.entries_arr, i64 0, i64 1), ptr null, ptr null }]
// SYCL-NEXT: @.sycl_offloading.descriptor = internal constant %__sycl.tgt_bin_desc { i16 1, i16 1, ptr @.sycl_offloading.device_images, ptr null, ptr null }

//      SYCL: define internal void @sycl.descriptor_reg() section ".text.startup" {
// SYCL-NEXT: entry:
// SYCL-NEXT:   call void @__sycl_register_lib(ptr @.sycl_offloading.descriptor)
// SYCL-NEXT:   ret void
// SYCL-NEXT: }

//      SYCL: define internal void @sycl.descriptor_unreg() section ".text.startup" {
// SYCL-NEXT: entry:
// SYCL-NEXT:   call void @__sycl_unregister_lib(ptr @.sycl_offloading.descriptor)
// SYCL-NEXT:   ret void
// SYCL-NEXT: }
