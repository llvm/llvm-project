// The driver forwards -emit-cir-bc to -cc1 like -emit-cir, stays in
// the compile phase, and names the default output after the cir-bc
// type.

// RUN: %clang -### -fclangir -emit-cir-bc %s 2>&1 | FileCheck %s
// CHECK: "-cc1"
// CHECK-SAME: "-emit-cir-bc"
// CHECK-SAME: "-o" "emit-cir-bc.cirbc"

// RUN: %clang -ccc-print-phases -fclangir -emit-cir-bc %s 2>&1 | FileCheck %s --check-prefix=PHASES
// PHASES: compiler, {{.*}}, cir-bc

// The flag implies the CIR pipeline without -fclangir.
// RUN: %clang -### -emit-cir-bc %s 2>&1 | FileCheck %s --check-prefix=NOFCLANGIR
// NOFCLANGIR: "-cc1"
// NOFCLANGIR-SAME: "-emit-cir-bc"

void foo() {}
