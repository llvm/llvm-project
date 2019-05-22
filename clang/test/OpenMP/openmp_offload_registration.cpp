// Test for offload registration code for two targets
// RUN: %clang_cc1 -verify -fopenmp -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-pc-linux-gnu,powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

void foo() {
#pragma omp target
  {}
}

// CHECK-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }
// CHECK-DAG: [[DEVTY:%.+]] = type { i8*, i8*, [[ENTTY]]*, [[ENTTY]]* }
// CHECK-DAG: [[DSCTY:%.+]] = type { i32, [[DEVTY]]*, [[ENTTY]]*, [[ENTTY]]* }

// Comdat key for the offload registration code. Should have sorted offload
// target triples encoded into the name.
// CHECK-DAG: $[[REGFN:\.omp_offloading\..+\.powerpc64le-ibm-linux-gnu\.x86_64-pc-linux-gnu+]] = comdat any

// Check if offloading descriptor is created.
// CHECK: [[ENTBEGIN:@.+]] = external constant [[ENTTY]]
// CHECK: [[ENTEND:@.+]] = external constant [[ENTTY]]
// CHECK: [[DEV1BEGIN:@.+]] = extern_weak constant i8
// CHECK: [[DEV1END:@.+]] = extern_weak constant i8
// CHECK: [[DEV2BEGIN:@.+]] = extern_weak constant i8
// CHECK: [[DEV2END:@.+]] = extern_weak constant i8
// CHECK: [[IMAGES:@.+]] = internal unnamed_addr constant [2 x [[DEVTY]]] [{{.+}} { i8* [[DEV1BEGIN]], i8* [[DEV1END]], [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }, {{.+}} { i8* [[DEV2BEGIN]], i8* [[DEV2END]], [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }], comdat($[[REGFN]])
// CHECK: [[DESC:@.+]] = internal constant [[DSCTY]] { i32 2, [[DEVTY]]* getelementptr inbounds ([2 x [[DEVTY]]], [2 x [[DEVTY]]]* [[IMAGES]], i32 0, i32 0), [[ENTTY]]* [[ENTBEGIN]], [[ENTTY]]* [[ENTEND]] }, comdat($[[REGFN]])

// Check target registration is registered as a Ctor.
// CHECK: appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @.omp_offloading.requires_reg, i8* null }, { i32, void ()*, i8* } { i32 0, void ()* @[[REGFN]], i8* bitcast (void ()* @[[REGFN]] to i8*) }]

// Check presence of foo() and the outlined target region
// CHECK: define void [[FOO:@.+]]()
// CHECK: define internal void [[OUTLINEDTARGET:@.+]]()

// Check registration and unregistration code.

// CHECK:     define internal void @.omp_offloading.requires_reg()
// CHECK:     call void @__tgt_register_requires(i64 1)
// CHECK:     ret void

// CHECK:     define internal void @[[UNREGFN:.+]](i8*)
// CHECK-SAME: comdat($[[REGFN]]) {
// CHECK:     call i32 @__tgt_unregister_lib([[DSCTY]]* [[DESC]])
// CHECK:     ret void
// CHECK:     declare i32 @__tgt_unregister_lib([[DSCTY]]*)

// CHECK:     define linkonce hidden void @[[REGFN]]()
// CHECK-SAME: comdat {
// CHECK:     call i32 @__tgt_register_lib([[DSCTY]]* [[DESC]])
// CHECK:     call i32 @__cxa_atexit(void (i8*)* @[[UNREGFN]], i8* bitcast ([[DSCTY]]* [[DESC]] to i8*),
// CHECK:     ret void
// CHECK:     declare i32 @__tgt_register_lib([[DSCTY]]*)

