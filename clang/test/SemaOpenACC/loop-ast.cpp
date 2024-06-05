
// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

void NormalFunc() {
  // CHECK-LABEL: NormalFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc loop
  for(;;);
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} <orphan>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt

  int array[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl
#pragma acc loop
  for(auto x : array){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} <orphan>
  // CHECK-NEXT: CXXForRangeStmt
  // CHECK: CompoundStmt

#pragma acc parallel
  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: CompoundStmt
  {
#pragma acc parallel
    // CHECK-NEXT: OpenACCComputeConstruct [[PAR_ADDR:[0-9a-fx]+]] {{.*}}parallel
    // CHECK-NEXT: CompoundStmt
    {
#pragma acc loop
      for(;;);
    // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: [[PAR_ADDR]]
    // CHECK-NEXT: ForStmt
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: NullStmt
    }
  }
}

template<typename T>
void TemplFunc() {
  // CHECK-LABEL: FunctionTemplateDecl {{.*}}TemplFunc
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: FunctionDecl{{.*}}TemplFunc
  // CHECK-NEXT: CompoundStmt

#pragma acc loop
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} <orphan>
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

#pragma acc parallel
  {
    // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
    // CHECK-NEXT: CompoundStmt
#pragma acc parallel
    {
    // CHECK-NEXT: OpenACCComputeConstruct [[PAR_ADDR_UNINST:[0-9a-fx]+]] {{.*}}parallel
    // CHECK-NEXT: CompoundStmt
#pragma acc loop
    // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: [[PAR_ADDR_UNINST]]
    // CHECK-NEXT: ForStmt
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: NullStmt
      for(;;);

#pragma acc loop
    // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: [[PAR_ADDR_UNINST]]
    // CHECK-NEXT: ForStmt
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: <<<NULL>>>
    // CHECK-NEXT: NullStmt
      for(;;);
    }
  }

  typename T::type array[5];
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl

#pragma acc loop
  for(auto x : array){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} <orphan>
  // CHECK-NEXT: CXXForRangeStmt
  // CHECK: CompoundStmt

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} TemplFunc 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'S'
  // CHECK-NEXT: RecordType{{.*}} 'S'
  // CHECK-NEXT: CXXRecord{{.*}} 'S'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} <orphan>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} used t 'typename S::type':'int'
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>
  // CHECK-NEXT: BinaryOperator{{.*}} 'bool' '<'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'typename S::type':'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'typename S::type':'int' lvalue Var
  // CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}} 'typename S::type':'int' lvalue prefix '++'
  // CHECK-NEXT: DeclRefExpr {{.*}} 'typename S::type':'int' lvalue Var
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} I 'typename S::type':'int'

  // CHECK-NEXT: OpenACCComputeConstruct {{.*}}parallel
  // CHECK-NEXT: CompoundStmt
  //
  // CHECK-NEXT: OpenACCComputeConstruct [[PAR_ADDR_INST:[0-9a-fx]+]] {{.*}}parallel
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: [[PAR_ADDR_INST]]
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} parent: [[PAR_ADDR_INST]]
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}} <orphan>
  // CHECK-NEXT: CXXForRangeStmt
  // CHECK: CompoundStmt
}

struct S {
  using type = int;
};

void use() {
  TemplFunc<S>();
}
#endif

