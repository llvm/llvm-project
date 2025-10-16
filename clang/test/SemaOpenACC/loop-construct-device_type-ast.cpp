// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s
#ifndef PCH_HELPER
#define PCH_HELPER

struct SomeS{};
void NormalUses() {
  // CHECK: FunctionDecl{{.*}}NormalUses
  // CHECK-NEXT: CompoundStmt

  SomeS SomeImpl;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} SomeImpl 'SomeS'
  // CHECK-NEXT: CXXConstructExpr
  bool SomeVar;
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} SomeVar 'bool'

#pragma acc loop device_type(default) dtype(nvidia)
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(default)
  // CHECK-NEXT: dtype(nvidia)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(acc_device_nvidia) dtype(radeon)
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(acc_device_nvidia)
  // CHECK-NEXT: dtype(radeon)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(host) dtype(multicore)
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(host)
  // CHECK-NEXT: dtype(multicore)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(default) dtype(*)
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(default)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(nvidia) dtype(*)
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(nvidia)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(radeon, host) dtype(*)
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(radeon, host)
  // CHECK-NEXT: dtype(*)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
}

template<typename T>
void TemplUses() {
  // CHECK-NEXT: FunctionTemplateDecl{{.*}}TemplUses
  // CHECK-NEXT: TemplateTypeParmDecl{{.*}}T
  // CHECK-NEXT: FunctionDecl{{.*}}TemplUses
  // CHECK-NEXT: CompoundStmt
#pragma acc loop device_type(default) dtype(multicore)
  for(int i = 0; i < 5; ++i){}
  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(default)
  // CHECK-NEXT: dtype(multicore)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt


  // Instantiations
  // CHECK-NEXT: FunctionDecl{{.*}} TemplUses 'void ()' implicit_instantiation
  // CHECK-NEXT: TemplateArgument type 'int'
  // CHECK-NEXT: BuiltinType{{.*}} 'int'
  // CHECK-NEXT: CompoundStmt

  // CHECK-NEXT: OpenACCLoopConstruct{{.*}}
  // CHECK-NEXT: device_type(default)
  // CHECK-NEXT: dtype(multicore)
  // CHECK-NEXT: ForStmt
  // CHECK-NEXT: DeclStmt
  // CHECK-NEXT: VarDecl{{.*}} i 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 0
  // CHECK-NEXT: <<<NULL>>>
  // CHECK-NEXT: BinaryOperator{{.*}}'<'
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 5
  // CHECK-NEXT: UnaryOperator{{.*}}++
  // CHECK-NEXT: DeclRefExpr{{.*}}'i' 'int'
  // CHECK-NEXT: CompoundStmt
}

void Inst() {
  TemplUses<int>();
}
#endif // PCH_HELPER
