// RUN: %clang_cc1 %s -fopenacc -Wno-openacc-cache-var-inside-loop -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void use() {
  // CHECK: FunctionDecl{{.*}} use 'void ()'
  // CHECK-NEXT: CompoundStmt
  int Array[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}Array 'int[5]'

#pragma acc loop
  // CHECK-NEXT: OpenACCLoopConstruct
  for (int i = 0; i < 5; ++i) {
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}i 'int'
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: UnaryOperator
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc cache(Array[1])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'Array' 'int[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
_Pragma("acc cache(Array[1])")
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'Array' 'int[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
#pragma acc cache(Array[1:2])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ArraySectionExpr{{.*}}'<array section type>' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'Array' 'int[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2

  // No vars here, since they are bad references.
#pragma acc cache (Array[i])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  }

  // No vars here, since they are bad references.
#pragma acc cache(Array[1:2])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache

  return;
  // CHECK-NEXT: ReturnStmt
}

struct S {
  int Array[5];
  int Array2D[5][5];

  void StructUse() {
    // CHECK: CXXMethodDecl{{.*}}StructUse 'void ()'
    // CHECK-NEXT: CompoundStmt
#pragma acc loop
  // CHECK-NEXT: OpenACCLoopConstruct
  for (int i = 0; i < 5; ++i) {
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}i 'int'
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: UnaryOperator
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc cache(Array[1])
    // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
    // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
    // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->Array
    // CHECK-NEXT: CXXThisExpr{{.*}} 'S *'
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
#pragma acc cache(Array[1:2])
    // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
    // CHECK-NEXT: ArraySectionExpr{{.*}}'<array section type>' lvalue
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
    // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->Array
    // CHECK-NEXT: CXXThisExpr{{.*}} 'S *'
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2
#pragma acc cache(Array2D[1][1])
    // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
    // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
    // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int[5]' lvalue
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)[5]' <ArrayToPointerDecay>
    // CHECK-NEXT: MemberExpr{{.*}} 'int[5][5]' lvalue ->Array2D
    // CHECK-NEXT: CXXThisExpr{{.*}} 'S *'
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
#pragma acc cache(Array2D[1][1:2])
    // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
    // CHECK-NEXT: ArraySectionExpr{{.*}}'<array section type>' lvalue
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
    // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int[5]' lvalue
    // CHECK-NEXT: ImplicitCastExpr{{.*}}'int (*)[5]' <ArrayToPointerDecay>
    // CHECK-NEXT: MemberExpr{{.*}} 'int[5][5]' lvalue ->Array2D
    // CHECK-NEXT: CXXThisExpr{{.*}} 'S *'
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
    // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2
  }
  }
};

template<typename T>
void templ_use() {
  // CHECK: FunctionDecl{{.*}} templ_use 'void ()'
  // CHECK-NEXT: CompoundStmt
  T Array[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}Array 'T[5]'

#pragma acc loop
  // CHECK-NEXT: OpenACCLoopConstruct
  for (int i = 0; i < 5; ++i) {
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}i 'int'
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: UnaryOperator
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: CompoundStmt
#pragma acc cache(Array[1])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'T' lvalue
  // CHECK-NEXT: DeclRefExpr{{.*}}'Array' 'T[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
#pragma acc cache(Array[1:2])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ArraySectionExpr{{.*}}'<dependent type>' lvalue
  // CHECK-NEXT: DeclRefExpr{{.*}}'Array' 'T[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2

    // No vars here, since they are bad references.
#pragma acc cache(Array[i])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  }
  // No vars here, since they are bad references.
#pragma acc cache(Array[1])
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache

  return;
  // CHECK-NEXT: ReturnStmt

  // Instantiation:
  // CHECK: FunctionDecl{{.*}} templ_use 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}Array 'int[5]'

  // CHECK-NEXT: OpenACCLoopConstruct
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}i 'int'
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: IntegerLiteral
  // CHECK-NEXT: UnaryOperator
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: CompoundStmt
  //
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'Array' 'int[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ArraySectionExpr{{.*}}'<array section type>' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'Array' 'int[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2

  // No vars here, since they are bad references.
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: OpenACCCacheConstruct{{.*}} cache
  // CHECK-NEXT: ReturnStmt
}

void foo() {
  templ_use<int>();
}
#endif
