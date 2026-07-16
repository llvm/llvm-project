// Test that -print-pipeline-passes also prints the codegen pipeline.

// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx900 \
// RUN:   -fenable-new-pm-codegen=force-on -emit-obj -o /dev/null \
// RUN:   -mllvm -print-pipeline-passes -O0 %s 2>&1 | FileCheck %s

// Don't try to check all passes, just a few codegen-specific ones (in order) to
// make sure the machine pipeline is actually printed.
// CHECK: require<MachineModuleAnalysis>
// CHECK-SAME: amdgpu-isel
// CHECK-SAME: prolog-epilog
// CHECK-SAME: amdgpu-asm-printer

void Foo(void) {}
