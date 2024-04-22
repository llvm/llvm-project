// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

int Global;
short GlobalArray[5];

void NormalUses(float *PointerParam) {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK: ParmVarDecl
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel private(Global, GlobalArray[2])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'short' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'short *' <ArrayToPointerDecay>
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 2
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(GlobalArray, PointerParam[Global])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(GlobalArray) private(PointerParam[Global])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'float' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'float *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}}'PointerParam' 'float *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(GlobalArray, PointerParam[Global : Global])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'float *' lvalue ParmVar{{.*}} 'PointerParam' 'float *'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

// This example is an error typically, but we want to make sure we're properly
// capturing NTTPs until instantiation time.
template<unsigned I>
void UnInstTempl() {
  // CHECK-NEXT: FunctionTemplateDecl{{.*}} UnInstTempl
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}}referenced 'unsigned int' depth 0 index 0 I
  // CHECK-NEXT: FunctionDecl{{.*}} UnInstTempl 'void ()'
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel private(I)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'unsigned int' NonTypeTemplateParm{{.*}}'I' 'unsigned int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

template<auto &NTTP, typename T, typename U>
void TemplUses(T t, U u, T*PointerParam) {
  // CHECK-NEXT: FunctionTemplateDecl
  // CHECK-NEXT: NonTypeTemplateParmDecl {{.*}}referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 1 T
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 2 U
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void (T, U, T *)'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced t 'T'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced u 'U'
  // CHECK-NEXT: ParmVarDecl{{.*}} referenced PointerParam 'T *'
  // CHECK-NEXT: CompoundStmt


#pragma acc parallel private(GlobalArray, PointerParam[Global])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'T' lvalue
  // CHECK-NEXT: DeclRefExpr{{.*}}'T *' lvalue ParmVar{{.*}}'PointerParam' 'T *'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(t, u)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(t) private(u)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(t) private(NTTP, u)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'auto' lvalue NonTypeTemplateParm{{.*}} 'NTTP' 'auto &'
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(u[0])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'<dependent type>' lvalue
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(u[0:t])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'U' lvalue ParmVar{{.*}} 'u' 'U'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: DeclRefExpr{{.*}}'T' lvalue ParmVar{{.*}} 't' 'T'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}EndMarker
  int EndMarker;

  // Check the instantiated versions of the above.
  // CHECK-NEXT: FunctionDecl{{.*}} used TemplUses 'void (int, int *, int *)' implicit_instantiation
  // CHECK-NEXT: TemplateArgument decl
  // CHECK-NEXT: Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: TemplateArgument type 'int[1]'
  // CHECK-NEXT: ConstantArrayType{{.*}} 'int[1]' 1
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used t 'int'
  // CHECK-NEXT: ParmVarDecl{{.*}} used u 'int *'
  // CHECK-NEXT: ParmVarDecl{{.*}} used PointerParam 'int *'
  // CHECK-NEXT: CompoundStmt

// #pragma acc parallel private(GlobalArray, PointerParam[Global])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'short[5]' lvalue Var{{.*}}'GlobalArray' 'short[5]'
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}}'PointerParam' 'int *'
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue Var{{.*}}'Global' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

// #pragma acc parallel private(t, u)
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

// #pragma acc parallel private(t) private(u)
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

// #pragma acc parallel private(t) private(NTTP, u)
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: private clause
  // CHECK-NEXT: SubstNonTypeTemplateParmExpr{{.*}}'const unsigned int' lvalue
  // CHECK-NEXT: NonTypeTemplateParmDecl{{.*}} referenced 'auto &' depth 0 index 0 NTTP
  // CHECK-NEXT: DeclRefExpr{{.*}}'const unsigned int' lvalue Var{{.*}} 'CEVar' 'const unsigned int'
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

// #pragma acc parallel private(u[0])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySubscriptExpr{{.*}}'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

// #pragma acc parallel private(u[0:t])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'int *' lvalue ParmVar{{.*}} 'u' 'int *'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: DeclRefExpr{{.*}}'int' lvalue ParmVar{{.*}} 't' 'int'
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}}EndMarker
}

struct S {
  // CHECK-NEXT: CXXRecordDecl{{.*}} struct S definition
  // CHECK: CXXRecordDecl{{.*}} implicit struct S
  int ThisMember;
  // CHECK-NEXT: FieldDecl{{.*}} ThisMember 'int'
  int ThisMemberArray[5];
  // CHECK-NEXT: FieldDecl{{.*}} ThisMemberArray 'int[5]'

  void foo();
  // CHECK-NEXT: CXXMethodDecl{{.*}} foo 'void ()'

  template<typename T>
  void bar() {
  // CHECK-NEXT: FunctionTemplateDecl{{.*}}bar
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}typename depth 0 index 0 T
  // CHECK-NEXT: CXXMethodDecl{{.*}} bar 'void ()' implicit-inline
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel private(ThisMember, this->ThisMemberArray[1])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: MemberExpr{{.*}} 'int' lvalue ->ThisMember
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' implicit this
  // CHECK-NEXT: ArraySubscriptExpr{{.*}} 'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ThisMemberArray[1:2])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr{{.*}}
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' implicit this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 2
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(this)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' this
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

  // Check Instantiations:
  // CHECK-NEXT: CXXMethodDecl{{.*}} used bar 'void ()' implicit_instantiation implicit-inline
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

// #pragma acc parallel private(ThisMember, this->ThisMemberArray[1])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: MemberExpr{{.*}} 'int' lvalue ->ThisMember
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' implicit this
  // CHECK-NEXT: ArraySubscriptExpr{{.*}} 'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

// #pragma acc parallel private(ThisMemberArray[1:2])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr{{.*}}
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' implicit this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 2
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

// #pragma acc parallel private(this)
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' this
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}
};

void S::foo() {
  // CHECK: CXXMethodDecl{{.*}} foo 'void ()'
  // CHECK-NEXT: CompoundStmt
#pragma acc parallel private(ThisMember, this->ThisMemberArray[1])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: MemberExpr{{.*}} 'int' lvalue ->ThisMember
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' implicit this
  // CHECK-NEXT: ArraySubscriptExpr{{.*}} 'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ThisMemberArray[1:2])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr{{.*}}
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' implicit this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 2
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(this)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: CXXThisExpr{{.*}} 'S *' this
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

template<typename U>
struct STempl {
  // CHECK-NEXT: ClassTemplateDecl{{.*}} STempl
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}} typename depth 0 index 0 U
  // CHECK-NEXT: CXXRecordDecl{{.*}} struct STempl definition
  // CHECK: CXXRecordDecl{{.*}} implicit struct STempl
  U ThisMember;
  // CHECK-NEXT: FieldDecl{{.*}} ThisMember 'U'
  U ThisMemberArray[5];
  // CHECK-NEXT: FieldDecl{{.*}} ThisMemberArray 'U[5]'

  void foo() {
    // CHECK-NEXT: CXXMethodDecl {{.*}} foo 'void ()'
    // CHECK-NEXT: CompoundStmt

#pragma acc parallel private(ThisMember, this->ThisMemberArray[1])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: MemberExpr{{.*}} 'U' lvalue ->ThisMember
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' implicit this
  // CHECK-NEXT: ArraySubscriptExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: CXXDependentScopeMemberExpr{{.*}} '<dependent type>' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ThisMemberArray[1:2])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr{{.*}}
  // CHECK-NEXT: MemberExpr{{.*}} 'U[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' implicit this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 2
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(this)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' this
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

  template<typename T>
  void bar() {
  // CHECK-NEXT: FunctionTemplateDecl{{.*}} bar
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}} typename depth 1 index 0 T
  // CHECK-NEXT: CXXMethodDecl{{.*}} bar 'void ()'
  // CHECK-NEXT: CompoundStmt

#pragma acc parallel private(ThisMember, this->ThisMemberArray[1])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: MemberExpr{{.*}} 'U' lvalue ->ThisMember
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' implicit this
  // CHECK-NEXT: ArraySubscriptExpr{{.*}} '<dependent type>' lvalue
  // CHECK-NEXT: CXXDependentScopeMemberExpr{{.*}} '<dependent type>' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(ThisMemberArray[1:2])
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr{{.*}}
  // CHECK-NEXT: MemberExpr{{.*}} 'U[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' implicit this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 2
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

#pragma acc parallel private(this)
  while(true);
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<U> *' this
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
}

// Instantiation of the class template.

// CHECK-NEXT: ClassTemplateSpecializationDecl{{.*}}struct STempl
// CHECK: TemplateArgument type 'int'
// CHECK-NEXT: BuiltinType {{.*}}'int'
// CHECK-NEXT: CXXRecordDecl{{.*}} struct STempl
// CHECK-NEXT: FieldDecl{{.*}}ThisMember 'int'
// CHECK-NEXT: FieldDecl{{.*}} ThisMemberArray 'int[5]'

// CHECK-NEXT: CXXMethodDecl{{.*}} foo 'void ()'
// CHECK-NEXT: FunctionTemplateDecl{{.*}}bar
// CHECK-NEXT: TemplateTypeParmDecl{{.*}} typename depth 0 index 0 T
// CHECK-NEXT: CXXMethodDecl{{.*}}bar 'void ()'
// CHECK-NEXT: CXXMethodDecl{{.*}}bar 'void ()'
// CHECK-NEXT: TemplateArgument type 'int'
// CHECK-NEXT: BuiltinType{{.*}} 'int'
// CHECK-NEXT: CompoundStmt

//#pragma acc parallel private(ThisMember, this->ThisMemberArray[1])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: MemberExpr{{.*}} 'int' lvalue ->ThisMember
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<int> *' implicit this
  // CHECK-NEXT: ArraySubscriptExpr{{.*}} 'int' lvalue
  // CHECK-NEXT: ImplicitCastExpr{{.*}} 'int *' <ArrayToPointerDecay>
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<int> *' this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(ThisMemberArray[1:2])
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: ArraySectionExpr{{.*}}
  // CHECK-NEXT: MemberExpr{{.*}} 'int[5]' lvalue ->ThisMemberArray
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<int> *' implicit this
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 1
  // CHECK-NEXT: IntegerLiteral{{.*}}'int' 2
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt

//#pragma acc parallel private(this)
  // CHECK-NEXT: OpenACCComputeConstruct{{.*}} parallel
  // CHECK-NEXT: private clause
  // CHECK-NEXT: CXXThisExpr{{.*}} 'STempl<int> *' this
  // CHECK-NEXT: WhileStmt
  // CHECK-NEXT: CXXBoolLiteralExpr
  // CHECK-NEXT: NullStmt
};

void Inst() {
  static constexpr unsigned CEVar = 1;
  int i;
  int Arr[5];
  TemplUses<CEVar, int, int[1]>({}, {}, &i);

  S s;
  s.bar<int>();
  STempl<int> stempl;
  stempl.bar<int>();
}
