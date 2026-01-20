// RUN: %clang_cc1 -E -std=c++20 %s

// CHECK: export __preprocessed_module M;
// CHECK-NEXT: export __preprocessed_import K;
// CHECK-NEXT: typedef int import;
// CHECK: import m;
export module M;
export import K;
typedef int import;
#define EMP
EMP import m;
