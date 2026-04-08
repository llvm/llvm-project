//  Test for debug info related to DW_AT_alignment attribute in the typedef operator
// Supported: -O0, standalone DI
// RUN: %clang_cc1 -dwarf-version=5  -emit-llvm -triple x86_64-linux-gnu %s -o - \
// RUN:   -O0 -disable-llvm-passes \
// RUN:   -debug-info-kind=standalone \
// RUN: | FileCheck %s

// CHECK: DIDerivedType(tag: DW_TAG_typedef, {{.*}}, align: 512

typedef char __attribute__((__aligned__(64))) alchar;

int main() {
  alchar newChar;
}
