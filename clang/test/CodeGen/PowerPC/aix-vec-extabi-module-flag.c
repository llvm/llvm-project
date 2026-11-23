// Check that the AIX extended Altivec ABI is emitted as the "target-abi" module
// flag

// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-feature +altivec \
// RUN:   -target-cpu pwr8 -mabi=vec-extabi -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=EXTABI
// RUN: %clang_cc1 -triple powerpc64-unknown-aix -target-feature +altivec \
// RUN:   -target-cpu pwr8 -mabi=vec-extabi -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=EXTABI

// RUN: %clang_cc1 -triple powerpc-unknown-aix -target-feature +altivec \
// RUN:   -target-cpu pwr8 -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=DFLTABI

// EXTABI: !{i32 1, !"target-abi", !"vec-extabi"}
// DFLTABI-NOT: "target-abi"

void f(void) {}
