// RUN: %clang_cc1 -fclangir -emit-cir -std=c99 %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -fclangir -emit-llvm -std=c99 %s -o - | FileCheck %s --check-prefix=LLVM_OGCG
// RUN: %clang_cc1 -emit-llvm -std=c99 %s -o - | FileCheck %s --check-prefix=LLVM_OGCG

inline void my_func() {}

// Force the externally visible definition
extern inline void my_func();

// CIR: module {{.*}} attributes {cir.lang = #cir.lang<c>{{.*}} {
// CIR-NEXT:   cir.func no_inline no_proto{{.*}}@my_func() attributes {{{.*}}, nothrow} {
// CIR-NEXT:     cir.return
// CIR-NEXT:   }
// CIR-NEXT: }

// LLVM_OGCG: define void @my_func(){{.*}}{
// LLVM_OGCG:   ret void
// LLVM_OGCG: }
