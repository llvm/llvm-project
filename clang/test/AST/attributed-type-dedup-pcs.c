// RUN: %clang_cc1 -triple arm-none-eabi -emit-pch -o %t.pch %s
// RUN: llvm-bcanalyzer --dump --disable-histogram %t.pch | FileCheck %s

// Pcs uses an EnumArgument and is encoded in the FunctionProtoType's
// calling convention bits on ARM targets, so the equivalent type
// discriminates aapcs vs aapcs-vfp. Two same-PCS uses must dedup; a
// different-PCS use must be distinct. Requires an ARM triple — on x86 /
// arm64-apple-darwin, pcs is silently ignored and no records are emitted.

void __attribute__((pcs("aapcs")))     f1(void);
void __attribute__((pcs("aapcs")))     f2(void);
void __attribute__((pcs("aapcs-vfp"))) f3(void);

// CHECK-COUNT-2: <TYPE_ATTRIBUTED
// CHECK-NOT:     <TYPE_ATTRIBUTED
