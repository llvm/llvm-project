// Enforcement recorded in a PCH must be restored and reach CodeGen, so a
// division in the main file traps just as it would with the attribute written
// inline (the enforce placement check is satisfied because the PCH precedes
// the main file).

// Test without pch.
// RUN: %clang_cc1 %s -fprofiles -std=c++23 -triple x86_64-linux-gnu -include %s -emit-llvm -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 %s -fprofiles -std=c++23 -triple x86_64-linux-gnu -emit-pch -o %t
// RUN: %clang_cc1 %s -fprofiles -std=c++23 -triple x86_64-linux-gnu -include-pch %t -emit-llvm -o - | FileCheck %s

#ifndef HEADER
#define HEADER

[[profiles::enforce(std::core_ub)]];

#else

// CHECK-LABEL: define {{.*}}@_Z6divideii
// CHECK: call void @llvm.ubsantrap(i8 3)
int divide(int a, int b) { return a / b; }

#endif
