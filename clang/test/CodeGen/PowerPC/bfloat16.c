// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
// RUN: %clang_cc1 -triple powerpc-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC32
// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -target-cpu pwr10 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
//
// Test that __bf16 is accepted on PowerPC targets and that the Clang
// frontend emits the expected bfloat IR type.

// __bf16 must be accepted (no "not supported on this target" error).
__bf16 global_bf;

// PPC64LE: @global_bf = global bfloat
// PPC32:   @global_bf = global bfloat

// Function signatures use bfloat type.
__bf16 add(__bf16 a, __bf16 b) {
  return a + b;
// PPC64LE-LABEL: define {{.*}} bfloat @add(bfloat noundef %a, bfloat noundef %b)
// PPC32-LABEL:   define {{.*}} bfloat @add(bfloat noundef %a, bfloat noundef %b)
}

__bf16 mul(__bf16 a, __bf16 b) {
  return a * b;
// PPC64LE-LABEL: define {{.*}} bfloat @mul(bfloat noundef %a, bfloat noundef %b)
// PPC32-LABEL:   define {{.*}} bfloat @mul(bfloat noundef %a, bfloat noundef %b)
}

// Extend/truncate round-trips.
float to_float(__bf16 a) {
  return (float)a;
// PPC64LE: fpext bfloat {{.*}} to float
// PPC32:   fpext bfloat {{.*}} to float
}

__bf16 from_float(float a) {
  return (__bf16)a;
// PPC64LE: fptrunc float {{.*}} to bfloat
// PPC32:   fptrunc float {{.*}} to bfloat
}

// sizeof and alignof must both be 2.
_Static_assert(sizeof(__bf16) == 2, "sizeof(__bf16) != 2");
_Static_assert(_Alignof(__bf16) == 2, "_Alignof(__bf16) != 2");
