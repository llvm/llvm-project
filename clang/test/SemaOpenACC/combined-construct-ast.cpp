// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel loop
  for(int i = 0; i < 5;++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}i
  // CHECK-NEXT: IntegerLiteral {{.*}}'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator {{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral {{.*}}'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} '++'
  // CHECK-NEXT: DeclRefExpr{{.*}}'i'
  // CHECK-NEXT: NullStmt

#pragma acc serial loop
  for(int i = 0; i < 5;++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} serial loop
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}i
  // CHECK-NEXT: IntegerLiteral {{.*}}'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator {{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral {{.*}}'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} '++'
  // CHECK-NEXT: DeclRefExpr{{.*}}'i'
  // CHECK-NEXT: NullStmt

#pragma acc kernels loop
  for(int i = 0; i < 5;++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} kernels loop
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}i
  // CHECK-NEXT: IntegerLiteral {{.*}}'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator {{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral {{.*}}'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} '++'
  // CHECK-NEXT: DeclRefExpr{{.*}}'i'
  // CHECK-NEXT: NullStmt


  int array[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl
#pragma acc parallel loop
  for(auto x : array){}
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}} parallel loop
  // CHECK-NEXT: CXXForRangeStmt
  // CHECK: CompoundStmt

#pragma acc parallel
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: CompoundStmt
  {
#pragma acc serial loop
    // CHECK-NEXT: OpenACCCombinedConstruct {{.*}}serial loop
    // CHECK-NEXT: CXXForRangeStmt
    // CHECK: CompoundStmt
    for(auto x: array) {
#pragma acc loop
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: serial loop
  // CHECK-NEXT: CXXForRangeStmt
  // CHECK: CompoundStmt
    for(auto y: array) { }
  }
  }
}
template<typename T>
void TemplFunc() {
  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc serial loop
  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}} serial loop
  for(typename T::type t = 0; t < 5;++t) {
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} referenced t 'typename T::type'
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: BinaryOperator{{.*}} '<dependent type>' '<'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'typename T::type' lvalue Var
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} '<dependent type>' lvalue prefix '++'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'typename T::type' lvalue Var
  // CHECK-NEXT: CompoundStmt
    typename T::type I;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename T::type'

  }

#pragma acc parallel loop
  for(int a = 0; a < 5; ++a) {
    // CHECK-NEXT: OpenACCCombinedConstruct {{.*}} parallel loop
    // CHECK-NEXT: ForStmt
    // CHECK: CompoundStmt
#pragma acc kernels loop
    for(int b = 0; b < 5; ++b) {
    // CHECK-NEXT: OpenACCCombinedConstruct {{.*}} kernels loop
    // CHECK-NEXT: ForStmt
    // CHECK: CompoundStmt
#pragma acc loop
    // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels loop
    // CHECK-NEXT: ForStmt
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl {{.*}} i 'int'
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
    // CHECK-NEXT: ImplicitCastExpr
    // CHECK-NEXT: DeclRefExpr
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
    // CHECK-NEXT: UnaryOperator{{.*}} prefix '++'
    // CHECK-NEXT: DeclRefExpr{{.*}} 'i' 'int'
    // CHECK-NEXT: NullStmt
      for(int i = 0; i < 5;++i);

#pragma acc loop
    // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels loop
    // CHECK-NEXT: ForStmt
    // CHECK-NEXT: DeclStmt
    // CHECK-NEXT: VarDecl {{.*}} i 'int'
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
    // CHECK-NEXT: ImplicitCastExpr
    // CHECK-NEXT: DeclRefExpr
    // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
    // CHECK-NEXT: UnaryOperator{{.*}} prefix '++'
    // CHECK-NEXT: DeclRefExpr{{.*}} 'i' 'int'
    // CHECK-NEXT: NullStmt
      for(int i = 0; i < 5;++i);
    }
  }

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'S'
  // CHECK-NEXT: RecordType{{.*}} 'S'
  // CHECK-NEXT: CXXRecord{{.*}} 'S'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}} serial loop
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} t 'typename S::type':'int'
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr {{.*}} 'typename S::type':'int' lvalue Var
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} 'typename S::type':'int' lvalue prefix '++'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'typename S::type':'int' lvalue Var
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename S::type':'int'

  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}} parallel loop
  // CHECK-NEXT: ForStmt
  // CHECK: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct {{.*}} kernels loop
  // CHECK-NEXT: ForStmt
  // CHECK: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels loop
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl {{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} prefix '++'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'i' 'int'
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: kernels loop
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl {{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} prefix '++'
  // CHECK-NEXT: DeclRefExpr{{.*}} 'i' 'int'
  // CHECK-NEXT: NullStmt

}

struct S {
  using type = int;
};

void use() {
  TemplFunc<S>();
}
#endif
