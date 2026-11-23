// This test checks that the implicit std::align_val_t is created with no source
// location and marked implicit when a new-expression triggers its synthesis.

// align_val_t is implicitly created in C++17+ (aligned allocation on by default).
// RUN: %clang_cc1 -std=c++17 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -ast-dump %s | FileCheck %s

// In older standards, -faligned-allocation must be explicit.
// RUN: %clang_cc1 -std=c++03 -faligned-allocation -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -faligned-allocation -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++14 -faligned-allocation -ast-dump %s | FileCheck %s

namespace std {}
void *p = new int;

// CHECK: NamespaceDecl {{.*}} std
// CHECK: EnumDecl {{.*}} <<invalid sloc>> <invalid sloc> implicit class align_val_t
