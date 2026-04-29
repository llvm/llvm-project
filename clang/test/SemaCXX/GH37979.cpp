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
    // CHECK-NEXT: CallExpr
    // CHECK-NEXT: ImplicitCastExpr{{.*}}<BuiltinFnToFnPtr>
    // CHECK-NEXT: DeclRefExpr{{.*}}__builtin_memcpy
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'void *' <BitCast>
    // CHECK-NEXT: UnaryOperator{{.*}} 'int *__restrict (*)[2]' prefix '&'
    // CHECK-NEXT: MemberExpr{{.*}} 'int *__restrict[2]' lvalue ->myPtr
    // CHECK-NEXT: CXXThisExpr{{.*}} 'Obj *' this
    //
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'const void *' <BitCast>
    // CHECK-NEXT: UnaryOperator{{.*}} 'int *__restrict const __restrict (*)[2]' prefix '&'
    // CHECK-NEXT: MemberExpr{{.*}} 'int *__restrict const __restrict[2]' lvalue .myPtr
    // CHECK-NEXT: DeclRefExpr{{.*}} 'const Obj' lvalue ParmVar
}
