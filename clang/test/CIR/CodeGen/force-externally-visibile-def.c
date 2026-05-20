// RUN: %clang_cc1 -fclangir -emit-cir -std=c99 %s -o - | FileCheck %s

inline void my_func() {}

// Force the externally visible definition
extern inline void my_func();

// CHECK: module {{.*}} attributes {cir.lang = #cir.lang<c>{{.*}} {
// CHECK-NEXT:   cir.func no_inline no_proto @my_func() attributes {{{.*}}, nothrow} {
// CHECK-NEXT:     cir.return
// CHECK-NEXT:   }
// CHECK-NEXT: }
