// RUN: %clang_cc1 -mllvm -emptyline-comment-coverage=false -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only %s | FileCheck %s

#define START_SCOPE {
#define END_SCOPE }

// CHECK: {{_Z2f0v|\?f0@@YAXXZ}}:
// CHECK-NEXT: File 0, [[@LINE+1]]:18 -> [[@LINE+1]]:20 = 0
inline void f0() {}

// CHECK: {{_Z2f1v|\?f1@@YAXXZ}}:
// CHECK-NEXT: File 0, [[@LINE+1]]:18 -> [[@LINE+1]]:31 = 0
inline void f1() START_SCOPE }

// CHECK: {{_Z2f2v|\?f2@@YAXXZ}}:
// CHECK-NEXT: File 0, [[@LINE+1]]:18 -> [[@LINE+1]]:29 = 0
inline void f2() { END_SCOPE

// CHECK: {{_Z2f3v|\?f3@@YAXXZ}}:
// CHECK-NEXT: File 0, [[@LINE+1]]:18 -> [[@LINE+1]]:39 = 0
inline void f3() START_SCOPE END_SCOPE

// CHECK: {{_Z2f4v|\?f4@@YAXXZ}}:
// CHECK-NEXT: File 0, [[@LINE+2]]:10 -> [[@LINE+3]]:2 = 0
inline void f4()
#include "Inputs/starts_a_scope_only"
}

// CHECK: {{_Z2f5v|\?f5@@YAXXZ}}:
// CHECK-NEXT: File 0, [[@LINE+1]]:18 -> [[@LINE+2]]:36 = 0
inline void f5() {
#include "Inputs/ends_a_scope_only"

// CHECK: {{_Z2f6v|\?f6@@YAXXZ}}:
// CHECK-NEXT: File 0, [[@LINE+2]]:10 -> [[@LINE+3]]:36 = 0
inline void f6()
#include "Inputs/starts_a_scope_only"
#include "Inputs/ends_a_scope_only"
