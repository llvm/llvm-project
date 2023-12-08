// Test that `-ffuchsia-api-level` is propagated to cc1.

// REQUIRES: x86-registered-target
// RUN: %clang -target x86_64-unknown-fuchsia -ffuchsia-api-level=16 -c %s -### 2>&1| FileCheck %s

// It should also be exposed to non-fuchsia platforms. This is desireable when
// using common Fuchsia headers for building host libraries that also depend on
// the Fuchsia version (such as using a compatible host-side FIDL library that
// talks with a Fuchsia FIDL library of the same version).
// RUN: %clang -target x86_64-unknown-linux-gnu -ffuchsia-api-level=16 -c %s -### 2>&1 | FileCheck %s

// Check Fuchsia API level macro.
// RUN: %clang -target x86_64-unknown-fuchsia -ffuchsia-api-level=15 -c %s -o %t
// RUN: llvm-readobj --symbols %t | FileCheck %s --check-prefix=CHECK-F15
//
// RUN: %clang -target x86_64-unknown-fuchsia -ffuchsia-api-level=16 -c %s -o %t
// RUN: llvm-readobj --symbols %t | FileCheck %s --check-prefix=CHECK-F16

// Check using a non-integer Fuchsia API level.
// RUN: not %clang -target x86_64-unknown-fuchsia -ffuchsia-api-level=16.0.0 -c %s  2>&1| FileCheck %s  --check-prefix=CHECK-ERROR


// CHECK: "-ffuchsia-api-level=16"

// CHECK-F15:   Name: f15

// CHECK-F16:   Name: f16

// CHECK-ERROR: error: invalid integral value '16.0.0' in '-ffuchsia-api-level=16.0.0'

#if __Fuchsia_API_level__ >= 16
void f16(void) {

}
#else
void f15(void) {

}
#endif

int main(int argc, char* argv[]) {
#if __Fuchsia_API_level__ >= 16
    f16();
#else
    f15();
#endif
}
