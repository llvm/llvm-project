// RUN: %clang_cc1 -triple arm-none-eabi -fdenormal-fp-math=positive-zero -emit-llvm -o - | FileCheck %s --check-prefix=DM-PZ
// RUN: %clang_cc1 -triple arm-none-eabi -fdenormal-fp-math=ieee -emit-llvm -o - | FileCheck %s --check-prefix=DM-IEEE
// RUN: %clang_cc1 -triple arm-none-eabi -fdenormal-fp-math=preserve-sign -emit-llvm -o - | FileCheck %s --check-prefix=DM-PS

// RUN: %clang_cc1 -triple arm-none-eabi -menable-no-infs -menable-no-nans -emit-llvm -o - | FileCheck %s --check-prefix=NM-FIN
// RUN: %clang_cc1 -triple arm-none-eabi -emit-llvm -o - | FileCheck %s --check-prefix=NM-IEEE

// RUN: %clang -target arm-none-eabi -S %s -emit-llvm -o - | FileCheck %s --check-prefix=FP-EXCEPT-IGNORE
// RUN: %clang -target arm-none-eabi -S %s -ffp-model=precise -emit-llvm -o - | FileCheck %s --check-prefix=FP-EXCEPT-IGNORE
// RUN: %clang -target arm-none-eabi -S %s -ffp-model=fast -emit-llvm -o - | FileCheck %s --check-prefix=FP-EXCEPT-IGNORE
// RUN: %clang -target arm-none-eabi -S %s -ffp-model=aggressive -emit-llvm -o - | FileCheck %s --check-prefix=FP-EXCEPT-IGNORE
// XUN: %clang -target arm-none-eabi -S %s -ffp-model=strict -emit-llvm -o - | FileCheck %s --check-prefix=FP-EXCEPT-CHECK

// DM-PZ: !{i32 7, !"arm-eabi-fp-denormal", i32 0}
// DM-IEEE: !{i32 4, !"arm-eabi-fp-denormal", i32 1}
// DM-PS: !{i32 7, !"arm-eabi-fp-denormal", i32 2}

// NM-FIN: !{i32 8, !"arm-eabi-fp-number-model", i32 1}
// NM-IEEE: !{i32 8, !"arm-eabi-fp-number-model", i32 3}

// FP-EXCEPT-IGNORE-NOT: !{i32 2, !"arm-eabi-fp-exceptions", i32 1}
// FP-EXCEPT-CHECK: !{i32 2, !"arm-eabi-fp-exceptions", i32 1}

void foo() {}
