// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER
void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

  int Val;

#pragma acc parallel loop vector
  for(int i = 0; i < 5; ++i);
  // CHECK: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop vector(Val)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop vector
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop vector
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop vector(Val)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}

template<typename T, unsigned One>
void TemplateUses(T Val) {
  // CHECK: FunctionTemplateDecl{{.*}}TemplateUses
  // CHECK-NEXT: TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'unsigned int' depth 0 index 1 One
  // CHECK-NEXT: FunctionDecl{{.*}} TemplateUses 'void (T)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced Val 'T'
  // CHECK-NEXT: CompoundStmt


#pragma acc parallel loop vector
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop vector(Val)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Val' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc parallel loop vector(length:One)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'One' 'unsigned int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc serial loop vector
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop vector
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop vector(Val)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'Val' 'T'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

#pragma acc kernels loop vector(length:One)
  for(int i = 0; i < 5; ++i);
  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'One' 'unsigned int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt


  // Instantiation:
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplateUses 'void (int)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: TemplateArgument integral '1U'
  // CHECK-NEXT: ParmVarDecl{{.*}} used Val 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}parallel loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr
  // CHECK-NEXT: NonTypeTemplateParmDecl
  // CHECK-NEXT: IntegerLiteral{{.*}}1
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}serial loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}} 'Val' 'int'
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt

  // CHECK-NEXT: OpenACCCombinedConstruct{{.*}}kernels loop
  // CHECK-NEXT: vector clause
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr
  // CHECK-NEXT: NonTypeTemplateParmDecl
  // CHECK-NEXT: IntegerLiteral{{.*}}1
  // CHECK-NEXT: ForStmt
  // CHECK: NullStmt
}

void inst() {
  TemplateUses<int, 1>(5);
}

#endif // PCH_HELPER
