// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s
// RUN: %clang_cc1 -fsyntax-only -ast-dump  %s | FileCheck %s
// expected-no-diagnostics

struct Obj { int * __restrict myPtr[2]; };

void do_copy() {
    Obj a, b;
    a = b;
    // CHECK-LABEL: CXXMethodDecl{{.*}} implicit used constexpr operator= 'Obj &(const Obj &) noexcept'
    // CHECK-NEXT: ParmVarDecl
    // CHECK-NEXT: CompoundStmt
    // Make sure that this uses the for-loop in the AST rather than trying to do
    // the early builtin_memcpy opt.
    // CHECK-NEXT: ForStmt
}
