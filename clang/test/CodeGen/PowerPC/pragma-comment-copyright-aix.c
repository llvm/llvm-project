// RUN: %clang_cc1 %s -triple powerpc-ibm-aix -O0 -disable-llvm-passes -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix -O0 -disable-llvm-passes -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple powerpc-ibm-aix -verify
// RUN: %clang_cc1 %s -triple powerpc64-ibm-aix -verify
// RUN: %clang_cc1 %s -DTEST_EMPTY_COPYRIGHT -triple powerpc-ibm-aix -verify

// RUN: %clang_cc1 %s -x c++ -triple powerpc-ibm-aix -O0 -disable-llvm-passes -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -x c++ -triple powerpc64-ibm-aix -O0 -disable-llvm-passes -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -x c++ -triple powerpc-ibm-aix -verify
// RUN: %clang_cc1 %s -x c++ -triple powerpc64-ibm-aix -verify
// RUN: %clang_cc1 %s -x c++ -DTEST_EMPTY_COPYRIGHT -triple powerpc-ibm-aix -verify

#ifndef TEST_EMPTY_COPYRIGHT
// Test basic pragma comment types
#pragma comment(copyright, "@(#) Copyright")

// Test duplicate copyright - should warn and ignore
#pragma comment(copyright, "Duplicate Copyright") // expected-warning {{'#pragma comment copyright' can be specified only once per translation unit - ignored}}

int main() { return 0; }

// Check that both metadata sections are present
// CHECK: !comment_string.loadtime = !{![[copyright:[0-9]+]]}

// Check individual metadata content
// CHECK: ![[copyright]] = !{!"@(#) Copyright"}

#else
// Test empty copyright string - valid with no warning
#pragma comment(copyright, "") // expected-no-diagnostics

int main() { return 0; }

#endif
