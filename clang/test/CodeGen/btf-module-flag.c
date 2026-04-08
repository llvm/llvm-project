// Verify that -gbtf sets the "BTF" module flag in LLVM IR.
//
// RUN: %clang_cc1 -triple x86_64-linux-gnu -gbtf -debug-info-kind=limited -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=BTF
// RUN: %clang_cc1 -triple x86_64-linux-gnu -debug-info-kind=limited -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefix=NO-BTF

// BTF: !{i32 2, !"BTF", i32 1}
// NO-BTF-NOT: !"BTF"

int main(void) { return 0; }
