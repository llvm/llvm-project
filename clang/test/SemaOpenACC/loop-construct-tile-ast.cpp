// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

struct S {
  constexpr S(){};
  constexpr operator auto() {return 1;}
};

void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

#pragma acc loop tile(S{}, 1, 2, *)
  for(;;)
    for(;;)
      for(;;)
        for(;;);
  // CHECK-NEXT: OpenACCLoopConstruct
  // CHECK-NEXT: tile clause
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator auto
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'S' lvalue
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}} 'S' 'void ()' list
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 2
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2
  // CHECK-NEXT: OpenACCAsteriskSizeExpr{{.*}}'int'
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt
}

template<typename T, unsigned Value>
void TemplUses() {
  // CHECK: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'unsigned int' depth 0 index 1 Value
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()'
  // CHECK-NEXT: CompoundStmt

#pragma acc loop tile(S{}, T{}, *, T{} + S{}, Value, Value + 3)
  for(;;)
    for(;;)
      for(;;)
        for(;;)
          for(;;)
            for(;;);
  // CHECK-NEXT: OpenACCLoopConstruct
  // CHECK-NEXT: tile clause
  //
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator auto
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'S' lvalue
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}} 'S' 'void ()' list
  //
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'T' 'T' list
  // CHECK-NEXT: InitListExpr{{.*}}'void'
  //
  // CHECK-NEXT: OpenACCAsteriskSizeExpr{{.*}}'int'
  //
  // CHECK-NEXT: BinaryOperator{{.*}}'<dependent type>' '+'
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}} 'T' 'T' list
  // CHECK-NEXT: InitListExpr{{.*}}'void'
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}} 'S' 'void ()' list
  //
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}}'Value' 'unsigned int'
  //
  // CHECK-NEXT: BinaryOperator{{.*}}'unsigned int' '+'
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}}'Value' 'unsigned int'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'unsigned int' <IntegralCast>
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 3
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt

  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'S'
  // CHECK-NEXT: RecordType{{.*}} 'S'
  // CHECK-NEXT: CXXRecord{{.*}} 'S'
  // CHECK-NEXT: TemplateArgument integral '2U'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct
  // CHECK-NEXT: tile clause
  //
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator auto
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'S' lvalue
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}} 'S' 'void ()' list
  //
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 1
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator auto
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'S' lvalue
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}} 'S' 'void ()' list
  //
  // CHECK-NEXT: OpenACCAsteriskSizeExpr{{.*}}'int'
  //
  // CHECK-NEXT: ConstantExpr{{.*}} 'int'
  // CHECK-NEXT: value: Int 2
  // CHECK-NEXT: BinaryOperator{{.*}}'int' '+'
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator auto
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'S' lvalue
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}} 'S' 'void ()' list
  // CHECK-NEXT: ImplicitCastExpr{{.*}}'int' <UserDefinedConversion>
  // CHECK-NEXT: CXXMemberCallExpr{{.*}}'int'
  // CHECK-NEXT: MemberExpr{{.*}} .operator auto
  // CHECK-NEXT: MaterializeTemporaryExpr{{.*}} 'S' lvalue
  // CHECK-NEXT: CXXTemporaryObjectExpr{{.*}} 'S' 'void ()' list
  //
  // CHECK-NEXT: ConstantExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: value: Int 2
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 Value
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 2
  //
  // CHECK-NEXT: ConstantExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: value: Int 5
  // CHECK-NEXT: BinaryOperator{{.*}}'unsigned int' '+'
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}} 'unsigned int'
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} 'unsigned int' depth 0 index 1 Value
  // CHECK-NEXT: IntegerLiteral{{.*}} 'unsigned int' 2
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'unsigned int' <IntegralCast>
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 3

  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: NullStmt

}

void Inst() {
  TemplUses<S, 2>();
}
#endif // PCH_HELPER
