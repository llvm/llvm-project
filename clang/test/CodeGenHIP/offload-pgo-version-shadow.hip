// NOTE: Do not autogenerate. Targeted checks express the relationships between
// global handles, registration calls, and entries retained by a two-TU link.
// REQUIRES: x86-registered-target
//
// Windows registers a per-TU host handle for the same module's raw version.
// Its eight-byte size is independent of the host pointer size.
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck %s --check-prefixes=WIN,STATIC --implicit-check-not=offloading.entry
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -cuid=abc -fgpu-rdc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck %s --check-prefixes=WIN,RDC
// RUN: %clang_cc1 -triple i686-pc-windows-msvc -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck %s --check-prefixes=WIN,STATIC
//
// Linux uses HSA collection, so it needs no version handle.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -cuid=abc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck %s --check-prefix=NO-VERSION
// RUN: %clang_cc1 -triple x86_64-linux-gnu -cuid=abc -fgpu-rdc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck %s --check-prefix=NO-VERSION
//
// No profile instrumentation or no CUID means no version registration.
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -cuid=abc \
// RUN:   -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck %s --check-prefix=NO-VERSION
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o - -x hip %s \
// RUN:   | FileCheck %s --check-prefix=NO-VERSION

// The shared device symbol must retain both per-TU host registrations at link.
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -cuid=abc -fgpu-rdc \
// RUN:   -fprofile-instrument=clang -emit-llvm -o %t.first.ll -x hip %s
// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -cuid=def -fgpu-rdc \
// RUN:   -DSECOND_TU -fprofile-instrument=clang -emit-llvm -o %t.second.ll -x hip %s
// RUN: llvm-link %t.first.ll %t.second.ll -S -o - | FileCheck %s --check-prefix=MULTI

#ifdef SECOND_TU
#define kernel second_kernel
#endif
extern "C" __attribute__((global)) void kernel(int *p) { *p += 1; }

// WIN: @__llvm_profile_sections_[[CUID:[0-9a-f]+]] = global ptr null
// WIN: @__llvm_profile_shadow_version_[[CUID]] = global i64 0
// STATIC: [[NAME:@[^ ]+]] = private unnamed_addr constant [27 x i8] c"__llvm_profile_raw_version\00"
// RDC: [[NAME:@[^ ]+]] = internal unnamed_addr constant [27 x i8] c"__llvm_profile_raw_version\00"
// RDC: @.offloading.entry.{{[^ ]+}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_shadow_version_[[CUID]], ptr [[NAME]], i64 8,
// WIN: @llvm.compiler.used = {{.*}}ptr @__llvm_profile_shadow_version_[[CUID]]
// STATIC-LABEL: define internal void @__hip_register_globals(
// STATIC: call void @__llvm_profile_offload_register_shadow_variable(ptr @__llvm_profile_sections_[[CUID]])
// STATIC-NEXT: call void @__hipRegisterVar({{.*}}ptr @__llvm_profile_shadow_version_[[CUID]], ptr [[NAME]], ptr [[NAME]], i32 0, i{{32|64}} 8, i32 0, i32 0)
// STATIC-NEXT: call void @__llvm_profile_offload_register_version_shadow_variable(ptr @__llvm_profile_shadow_version_[[CUID]])
// RDC-LABEL: define internal void @__llvm_profile_register_shadow.
// RDC-SAME: [[CUID]]()
// RDC: call void @__llvm_profile_offload_register_shadow_variable(ptr @__llvm_profile_sections_[[CUID]])
// RDC-NEXT: call void @__llvm_profile_offload_register_version_shadow_variable(ptr @__llvm_profile_shadow_version_[[CUID]])
// WIN: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_data_[[CUID]]_0)
// WIN: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_cnts_[[CUID]]_1)
// WIN: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_ucnts_[[CUID]]_2)
// WIN: call void @__llvm_profile_offload_register_section_shadow_variable(ptr @__llvm_profile_shadow_names_[[CUID]]_3)

// NO-VERSION-NOT: __llvm_profile_shadow_version_
// NO-VERSION-NOT: __llvm_profile_offload_register_version_shadow_variable

// MULTI-DAG: @__llvm_profile_shadow_version_[[FIRST:[0-9a-f]+]] = global i64 0
// MULTI-DAG: @__llvm_profile_shadow_version_[[SECOND:[0-9a-f]+]] = global i64 0
// MULTI-DAG: @.offloading.entry.{{[^ ]+}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_shadow_version_[[FIRST]], {{.*}}i64 8,
// MULTI-DAG: @.offloading.entry.{{[^ ]+}} = weak constant %struct.__tgt_offload_entry {{.*}}ptr @__llvm_profile_shadow_version_[[SECOND]], {{.*}}i64 8,
// MULTI-DAG: call void @__llvm_profile_offload_register_version_shadow_variable(ptr @__llvm_profile_shadow_version_[[FIRST]])
// MULTI-DAG: call void @__llvm_profile_offload_register_version_shadow_variable(ptr @__llvm_profile_shadow_version_[[SECOND]])
