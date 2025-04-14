// Test that the accelerator code selection pass only gets invoked after linking

// Ensure Pass HipStdParAcceleratorCodeSelectionPass is not invoked in PreLink.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -mllvm -amdgpu-enable-hipstdpar -flto -emit-llvm-bc -fcuda-is-device -fdebug-pass-manager \
// RUN:  %s -o /dev/null 2>&1 | FileCheck --check-prefix=HIPSTDPAR-PRE %s
// HIPSTDPAR-PRE: Running pass: EntryExitInstrumenterPass
// HIPSTDPAR-PRE-NEXT: Running pass: EntryExitInstrumenterPass
// HIPSTDPAR-PRE-NOT: Running pass: HipStdParAcceleratorCodeSelectionPass
// HIPSTDPAR-PRE-NEXT: Running pass: AlwaysInlinerPass

// Ensure Pass HipStdParAcceleratorCodeSelectionPass is invoked in PostLink.
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -mllvm -amdgpu-enable-hipstdpar -fcuda-is-device -fdebug-pass-manager -emit-llvm \
// RUN:  %s -o /dev/null 2>&1 | FileCheck --check-prefix=HIPSTDPAR-POST %s
// HIPSTDPAR-POST: Running pass: HipStdParAcceleratorCodeSelection

#define __device__ __attribute__((device))

void foo(float *a, float b) {
  *a = b;
}

__device__ void bar(float *a, float b) {
  *a = b;
}
