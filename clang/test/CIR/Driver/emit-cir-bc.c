// -emit-cir follows the -emit-llvm model: -c selects the binary form,
// bare -emit-cir and -emit-cir -S keep emitting text. -emit-cir-bc is
// cc1-only.

// RUN: %clang -### -fclangir -emit-cir -c %s 2>&1 | FileCheck %s --check-prefix=BC
// BC: "-cc1"
// BC-SAME: "-emit-cir-bc"
// BC-SAME: "-o" "emit-cir-bc.cirbc"

// RUN: %clang -ccc-print-phases -fclangir -emit-cir -c %s 2>&1 | FileCheck %s --check-prefix=PHASES
// PHASES: compiler, {{.*}}, cir-bc

// RUN: %clang -### -fclangir -emit-cir %s 2>&1 | FileCheck %s --check-prefix=TEXT
// RUN: %clang -### -fclangir -emit-cir -S %s 2>&1 | FileCheck %s --check-prefix=TEXT
// TEXT: "-cc1"
// TEXT-SAME: "-emit-cir"
// TEXT-SAME: "-o" "emit-cir-bc.cir"

// RUN: not %clang -fclangir -emit-cir-bc %s 2>&1 | FileCheck %s --check-prefix=REJECT
// REJECT: unknown argument '-emit-cir-bc'

// The driver accepts .cirbc inputs, so bytecode round-trips through it.
// RUN: %clang -fclangir -emit-cir -c %s -o %t.cirbc
// RUN: %clang -fclangir -x cir-bc -emit-cir %t.cirbc -o - | FileCheck %s --check-prefix=ROUNDTRIP
// RUN: %clang -fclangir -emit-cir %t.cirbc -o - | FileCheck %s --check-prefix=ROUNDTRIP
// ROUNDTRIP: cir.func {{.*}}@foo()

void foo() {}
