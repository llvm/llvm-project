// Checks encoding of output file
// This is only required for z/OS.
//
// REQUIRES: system-zos, systemz-registered-target
// RUN: %clang_cc1 -triple s390x-ibm-zos -S %s -o %t.s
// RUN: ls -T %t.s | FileCheck %s

// CHECK: t IBM-1047    T=on
void foo() { return; }
