// Normally this file can't be compiled due to
// nullptr cast in offset calculation
// RUN: not %clang_cc1 -ast-dump  %s -o - 2>&1 | FileCheck %s --check-prefix=AST-ORIG
// Kernel variant works OK.
// RUN: %clang_cc1 -fms-kernel -ast-dump  %s -o - | FileCheck %s --check-prefix=AST-NEW

// AST-ORIG: error: constexpr function never produces a constant expression [-Winvalid-constexpr]
// AST-ORIG: FunctionDecl
// AST-ORIG-NEXT: ParmVarDecl
// AST-ORIG-NEXT: CompoundStmt
// AST-ORIG-NEXT: ReturnStmt
// AST-ORIG-NEXT: ParenExpr
// AST-ORIG-NEXT: CStyleCastExpr
// AST-ORIG-NEXT: ParenExpr
// AST-ORIG-NEXT: BinaryOperator
// AST-ORIG-NEXT: CStyleCastExpr
// AST-ORIG-NEXT: ImplicitCastExpr
// AST-ORIG-NEXT: ParenExpr
// AST-ORIG-NEXT: DeclRefExpr
// AST-ORIG-NEXT: CStyleCastExpr
// AST-ORIG-NEXT: ParenExpr
// AST-ORIG-NEXT: UnaryOperator
// AST-ORIG-NEXT: MemberExpr
// AST-ORIG-NEXT: ParenExpr
// AST-ORIG-NEXT: CStyleCastExpr
// AST-ORIG-NEXT: ImplicitCastExpr
// AST-ORIG-NEXT: IntegerLiteral

// AST-NEW: FunctionDecl
// AST-NEW-NEXT: ParmVarDecl
// AST-NEW-NEXT: CompoundStmt
// AST-NEW-NEXT: ReturnStmt
// AST-NEW-NEXT: ParenExpr
// AST-NEW-NEXT: CStyleCastExpr
// AST-NEW-NEXT: ParenExpr
// AST-NEW-NEXT: BinaryOperator
// AST-NEW-NEXT: CStyleCastExpr
// AST-NEW-NEXT: ImplicitCastExpr
// AST-NEW-NEXT: ParenExpr
// AST-NEW-NEXT: DeclRefExpr
// AST-NEW-NEXT: CStyleCastExpr
// AST-NEW-NEXT: ImplicitCastExpr
// AST-NEW-NEXT: OffsetOfExpr

typedef char* PCHAR;
typedef unsigned long long ULONG_PTR;

#define CONTAINING_RECORD_UB(address, type, field) ((type *)( \
    (PCHAR)(address) - (ULONG_PTR)(&((type *)0)->field)))


struct MyStruct { int a; int b; int c; };

constexpr MyStruct* get_container_ub(int* p) {
    return CONTAINING_RECORD_UB(p, MyStruct, b);
}

