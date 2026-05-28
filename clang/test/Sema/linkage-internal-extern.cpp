// RUN: %clang_cc1 -ast-dump -std=c++17 %s | FileCheck %s

// In C++, block-scope extern declarations target the enclosing namespace
// scope ([dcl.meaning.general]/3.5), so they match against the namespace-scope
// static despite local shadows and inherit internal linkage. No conflict arises.
//
// This differs from C, where a local shadow breaks linkage inheritance,
// causing the conflict diagnosed by err_internal_extern_mismatch.

// Example adapted from [basic.link]/6.
static void f();
// CHECK: FunctionDecl {{.*}} f 'void ()' static internal-linkage
static int i = 0;
// CHECK: VarDecl {{.*}} i 'int' static cinit internal-linkage
void g() {
// CHECK: FunctionDecl {{.*}} g 'void ()' external-linkage
    extern void f();
    // CHECK: FunctionDecl {{.*}} prev {{.*}} f 'void ()' extern internal-linkage
    int i;
    // CHECK: VarDecl {{.*}} i 'int'{{$}}
    {
        extern void f();
        // CHECK: FunctionDecl {{.*}} prev {{.*}} f 'void ()' extern internal-linkage
        extern int i;
        // CHECK: VarDecl {{.*}} prev {{.*}} i 'int' extern internal-linkage
    }
}

// Block-scope function declarations behave identically without extern
// (C11 6.2.2p5, C++ [dcl.meaning.general]/3.5).
static void h();
// CHECK: FunctionDecl {{.*}} h 'void ()' static internal-linkage
void g2() {
// CHECK: FunctionDecl {{.*}} g2 'void ()' external-linkage
    int h;
    // CHECK: VarDecl {{.*}} h 'int'{{$}}
    {
        void h();
        // CHECK: FunctionDecl {{.*}} prev {{.*}} h 'void ()' internal-linkage
    }
}
