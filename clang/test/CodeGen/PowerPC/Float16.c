// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
// RUN: %clang_cc1 -triple powerpc-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC32
// RUN: %clang_cc1 -triple powerpc64le-linux-gnu -target-cpu pwr9 -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=PPC64LE
//
// Test that _Float16 is accepted on PowerPC targets and that the Clang
// frontend emits the expected 'half' IR type for _Float16 arithmetic.

// _Float16 must be accepted (no "not supported on this target" error).
_Float16 global_h = 1.0f16;

// PPC64LE: @global_h = global half
// PPC32:   @global_h = global half

// Function signatures use 'half' type.
_Float16 add(_Float16 a, _Float16 b) {
  return a + b;
// PPC64LE-LABEL: define {{.*}} half @add(half noundef %a, half noundef %b)
// PPC32-LABEL:   define {{.*}} half @add(half noundef %a, half noundef %b)
}

_Float16 mul(_Float16 a, _Float16 b) {
  return a * b;
// PPC64LE-LABEL: define {{.*}} half @mul(half noundef %a, half noundef %b)
// PPC32-LABEL:   define {{.*}} half @mul(half noundef %a, half noundef %b)
}

// Extend/truncate round-trips.
float to_float(_Float16 a) {
  return (float)a;
// PPC64LE: fpext half {{.*}} to float
// PPC32:   fpext half {{.*}} to float
}

_Float16 from_float(float a) {
  return (_Float16)a;
// PPC64LE: fptrunc float {{.*}} to half
// PPC32:   fptrunc float {{.*}} to half
}

// sizeof must be 2 and _Alignof must be 2.
_Static_assert(sizeof(_Float16) == 2, "sizeof(_Float16) != 2");
_Static_assert(_Alignof(_Float16) == 2, "_Alignof(_Float16) != 2");
