// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

auto Lambda = [](){};
#pragma acc routine(Lambda) worker nohost bind("string")
// CHECK: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' '(lambda at
// CHECK-NEXT: worker clause
// CHECK-NEXT: nohost clause
// CHECK-NEXT: bind clause
// CHECK-NEXT: StringLiteral{{.*}} "string"

int function();
#pragma acc routine (function) nohost vector bind(identifier)
// CHECK: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'function' 'int ()'
// CHECK-NEXT: nohost clause
// CHECK-NEXT: vector clause
// CHECK-NEXT: bind clause identifier 'identifier'

#pragma acc routine(function) device_type(Something) seq
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'function' 'int ()'
// CHECK-NEXT: device_type(Something)
// CHECK-NEXT: seq clause
#pragma acc routine(function) nohost dtype(Something) vector
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'function' 'int ()'
// CHECK-NEXT: nohost clause
// CHECK-NEXT: dtype(Something)
// CHECK-NEXT: vector clause

#pragma acc routine nohost vector
int function2();
// CHECK: FunctionDecl{{.*}} function2
// CHECK-NEXT: OpenACCRoutineDeclAttr
// CHECK-NEXT: nohost clause
// CHECK-NEXT: vector clause

#pragma acc routine worker nohost bind("asdf")
auto Lambda2 = [](){};
// CHECK: VarDecl{{.*}} Lambda2 '(lambda at
// CHECK: CXXMethodDecl{{.*}} operator()
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: OpenACCRoutineDeclAttr
// CHECK-NEXT: worker clause
// CHECK-NEXT: nohost clause
// CHECK-NEXT: bind clause
// CHECK-NEXT: StringLiteral{{.*}}"asdf"

#pragma acc routine worker nohost bind("asdf")
auto Lambda3 = [](auto a){ return a;};
// CHECK: VarDecl{{.*}} Lambda3 '(lambda at
// CHECK-NEXT: LambdaExpr
// CHECK-NEXT: CXXRecordDecl
// CHECK: FunctionTemplateDecl{{.*}} operator()
// CHECK-NEXT: TemplateTypeParmDecl
// CHECK-NEXT: CXXMethodDecl{{.*}} operator() 'auto (auto) const'
// CHECK-NEXT: ParmVarDecl{{.*}}'auto'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: DeclRefExpr
// CHECK: OpenACCRoutineDeclAttr
// CHECK-NEXT: worker clause
// CHECK-NEXT: nohost clause
// CHECK-NEXT: bind clause
// CHECK-NEXT: StringLiteral{{.*}}"asdf"
// CHECK-NEXT: CXXMethodDecl{{.*}}operator() 'int (int) const'{{.*}} implicit_instantiation
// CHECK-NEXT: TemplateArgument type 'int'
// CHECK-NEXT: BuiltinType
// CHECK-NEXT: ParmVarDecl{{.*}}'int'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr
// CHECK-NEXT: OpenACCRoutineDeclAttr
// CHECK-NEXT: worker clause
// CHECK-NEXT: nohost clause
// CHECK-NEXT: bind clause
// CHECK-NEXT: StringLiteral{{.*}}"asdf"

int InstL3 = Lambda3(1);

namespace NS {
  int NSFunc();
auto Lambda = [](){};
}
#pragma acc routine(NS::NSFunc) seq
// CHECK: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'NSFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'NS'
// CHECK-NEXT: seq clause
#pragma acc routine(NS::Lambda) gang
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'NS::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'NS'
// CHECK-NEXT: gang clause

constexpr int getInt() { return 1; }

struct S {
  void MemFunc();
#pragma acc routine gang(dim: 1)
  void MemFunc2();
  // CHECK: CXXMethodDecl{{.*}}MemFunc2
  // CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}} 
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 1
  static void StaticMemFunc();
#pragma acc routine gang(dim: getInt())
  static void StaticMemFunc2();
  // CHECK: CXXMethodDecl{{.*}}StaticMemFunc2
  // CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: ConstantExpr{{.*}}'int'
  // CHECK-NEXT: value: Int 1
  constexpr static auto Lambda = [](){ return 1; };
#pragma acc routine worker
  constexpr static auto Lambda2 = [](){ return 1; };
  // CHECK: VarDecl{{.*}}Lambda2 'const S::(lambda at
  // CHECK-NEXT: value: Struct
  // CHECK-NEXT: ImplicitCastExpr
  // CHECK-NEXT: LambdaExpr
  // CHECK-NEXT: CXXRecordDecl
  // CHECK: CXXMethodDecl{{.*}} operator() 'int () const'
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: ReturnStmt
  // CHECK-NEXT: IntegerLiteral
  // CHECK: OpenACCRoutineDeclAttr{{.*}}
  // CHECK-NEXT: worker clause

#pragma acc routine(S::MemFunc) gang(dim: 1)
// CHECK: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
#pragma acc routine(S::StaticMemFunc) gang(dim:getInt())
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
#pragma acc routine(S::Lambda) worker
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: worker clause

#pragma acc routine(MemFunc) gang(dim: 1)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
#pragma acc routine(StaticMemFunc) gang(dim:Lambda())
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
#pragma acc routine(Lambda) worker
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: worker clause
#pragma acc routine(Lambda) worker device_type(Lambda)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: worker clause
// CHECK-NEXT: device_type(Lambda)
#pragma acc routine(Lambda) dtype(Lambda) vector
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: dtype(Lambda)
// CHECK-NEXT: vector clause
};

#pragma acc routine(S::MemFunc) gang(dim: 1)
// CHECK: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
#pragma acc routine(S::StaticMemFunc) worker
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: worker clause
#pragma acc routine(S::Lambda) vector
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: vector

template<typename T>
struct DepS {
  T MemFunc();
  static T StaticMemFunc();
#pragma acc routine gang(dim: T{1})
  static T StaticMemFunc2();
  // CHECK: CXXMethodDecl{{.*}}StaticMemFunc2
  // CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'T' list
  constexpr static auto Lambda = [](){return 1;};
#pragma acc routine gang(dim: T{1})
  constexpr static auto Lambda2 = [](){return 1;};
  // CHECK: VarDecl{{.*}}Lambda2 'const auto'
  // CHECK-NEXT: LambdaExpr
  // CHECK-NEXT: CXXRecordDecl
  // CHECK: CXXMethodDecl{{.*}} operator()
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: ReturnStmt
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'T' list
#pragma acc routine gang(dim: T{1})
  constexpr static auto Lambda3 = [](auto){return 1;};
  // CHECK: VarDecl{{.*}}Lambda3 'const auto'
  // CHECK-NEXT: LambdaExpr
  // CHECK-NEXT: CXXRecordDecl
  // CHECK: FunctionTemplateDecl{{.*}} operator()
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: CXXMethodDecl{{.*}} operator() 'auto (auto) const'
  // CHECK-NEXT: ParmVarDecl{{.*}}'auto'
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: ReturnStmt
  // CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
  // CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: CXXUnresolvedConstructExpr{{.*}}'T' list
#pragma acc routine gang(dim: Lambda())
  T MemFunc2();
  // CHECK: CXXMethodDecl{{.*}}MemFunc2
  // CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
  // CHECK-NEXT: gang clause
  // CHECK-NEXT: CallExpr{{.*}}'<dependent type>'

#pragma acc routine(Lambda) gang(dim: Lambda())
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const auto'
// CHECK-NEXT: gang clause
// CHECK-NEXT: CallExpr{{.*}}'<dependent type>'
#pragma acc routine(MemFunc) worker
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'T ()'
// CHECK-NEXT: worker clause
#pragma acc routine(StaticMemFunc) seq
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'T ()'
// CHECK-NEXT: seq clause

#pragma acc routine(DepS::Lambda) gang(dim:1)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const auto'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
#pragma acc routine(DepS::MemFunc) gang
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
// CHECK-NEXT: gang clause
#pragma acc routine(DepS::StaticMemFunc) worker
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
// CHECK-NEXT: worker clause

#pragma acc routine(DepS<T>::Lambda) vector
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const auto'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
// CHECK-NEXT: vector clause
#pragma acc routine(DepS<T>::MemFunc) seq
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
// CHECK-NEXT: seq clause
#pragma acc routine(DepS<T>::StaticMemFunc) worker
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
// CHECK-NEXT: worker clause
#pragma acc routine(DepS<T>::StaticMemFunc) worker device_type(T)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
// CHECK-NEXT: worker clause
// CHECK-NEXT: device_type(T)

// Instantiation:
// CHECK: ClassTemplateSpecializationDecl{{.*}}struct DepS
// CHECK: CXXRecordDecl{{.*}} struct DepS

// CHECK: CXXMethodDecl{{.*}}StaticMemFunc2
// CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1

// CHECK: VarDecl{{.*}}Lambda2 'const DepS<int>::(lambda at
// CHECK-NEXT: value: Struct
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: LambdaExpr
// CHECK-NEXT: CXXRecordDecl
// CHECK: CXXMethodDecl{{.*}} operator()
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
// CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1

// CHECK: VarDecl{{.*}}Lambda3 'const DepS<int>::(lambda at
// CHECK-NEXT: value: Struct
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: LambdaExpr
// CHECK-NEXT: CXXRecordDecl
// CHECK: FunctionTemplateDecl{{.*}} operator()
// CHECK-NEXT: TemplateTypeParmDecl
// CHECK-NEXT: CXXMethodDecl{{.*}} operator() 'auto (auto) const'
// CHECK-NEXT: ParmVarDecl{{.*}}'auto'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: ReturnStmt
// CHECK-NEXT: IntegerLiteral{{.*}} 'int' 1
// CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1

// CHECK: CXXMethodDecl{{.*}}MemFunc2
// CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1

// CHECK: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'
// CHECK-NEXT: worker clause

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: seq clause

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: gang clause

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: worker clause

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: vector clause 

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: seq clause 

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: worker clause

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: worker clause
// CHECK-NEXT: device_type(T)
};

#pragma acc routine(DepS<int>::Lambda) gang(dim:1)
// CHECK: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1
#pragma acc routine(DepS<int>::MemFunc) worker
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: worker clause
#pragma acc routine(DepS<int>::StaticMemFunc) vector
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
// CHECK-NEXT: vector clause

template<typename T>
void TemplFunc() {
#pragma acc routine(T::MemFunc) gang(dim:T::Lambda())
// CHECK: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
// CHECK-NEXT: gang clause
// CHECK-NEXT: CallExpr{{.*}}'<dependent type>'
#pragma acc routine(T::StaticMemFunc) nohost worker bind("string")
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
// CHECK-NEXT: nohost clause
// CHECK-NEXT: worker clause
// CHECK-NEXT: bind clause
// CHECK-NEXT: StringLiteral{{.*}} "string"
#pragma acc routine(T::Lambda) seq nohost bind(identifier)
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
// CHECK-NEXT: seq clause
// CHECK-NEXT: nohost clause
// CHECK-NEXT: bind clause identifier 'identifier'

 auto Lambda1 = [](){};
#pragma acc routine(Lambda1) seq
// CHECK: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda1' 'auto'
// CHECK-NEXT: seq clause

#pragma acc routine seq
  auto Lambda2 = [](){};
  // CHECK: VarDecl{{.*}} Lambda2 'auto'
  // CHECK-NEXT: LambdaExpr
  // CHECK-NEXT: CXXRecordDecl
  // CHECK: CXXMethodDecl{{.*}} operator()
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
  // CHECK-NEXT: seq clause

#pragma acc routine seq
  auto Lambda3 = [](auto){};
  // CHECK: VarDecl{{.*}} Lambda3 'auto'
  // CHECK-NEXT: LambdaExpr
  // CHECK-NEXT: CXXRecordDecl
  // CHECK: FunctionTemplateDecl{{.*}} operator()
  // CHECK-NEXT: TemplateTypeParmDecl
  // CHECK-NEXT: CXXMethodDecl{{.*}} operator() 'auto (auto) const'
  // CHECK-NEXT: ParmVarDecl{{.*}}'auto'
  // CHECK-NEXT: CompoundStmt
  // CHECK-NEXT: OpenACCRoutineDeclAttr
  // CHECK-NEXT: seq clause

  Lambda3(T{});

// Instantiation:
// CHECK: FunctionDecl{{.*}} TemplFunc 'void ()' implicit_instantiation
// CHECK: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}}
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: gang clause
// CHECK-NEXT: ConstantExpr{{.*}}'int'
// CHECK-NEXT: value: Int 1

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}}
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: nohost clause
// CHECK-NEXT: worker clause
// CHECK-NEXT: bind clause
// CHECK-NEXT: StringLiteral{{.*}} "string"

// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}}
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: seq clause
// CHECK-NEXT: nohost clause
// CHECK-NEXT: bind clause identifier 'identifier'

// CHECK: OpenACCRoutineDecl{{.*}} routine 
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda1' '(lambda at
// CHECK-NEXT: seq clause

// CHECK: VarDecl{{.*}} Lambda2 '(lambda at
// CHECK-NEXT: LambdaExpr
// CHECK-NEXT: CXXRecordDecl
// CHECK: CXXMethodDecl{{.*}} operator()
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: OpenACCRoutineDeclAttr{{.*}}
// CHECK-NEXT: seq clause

// CHECK: VarDecl{{.*}} Lambda3 '(lambda at
// CHECK-NEXT: LambdaExpr
// CHECK-NEXT: CXXRecordDecl
// CHECK: FunctionTemplateDecl{{.*}} operator()
// CHECK-NEXT: TemplateTypeParmDecl
// CHECK-NEXT: CXXMethodDecl{{.*}} operator() 'auto (auto) const'
// CHECK-NEXT: ParmVarDecl{{.*}}'auto'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: OpenACCRoutineDeclAttr
// CHECK-NEXT: seq clause
// CHECK: CXXMethodDecl{{.*}} operator() 'void (S) const'
// CHECK-NEXT: TemplateArgument
// CHECK-NEXT: RecordType
// CHECK-NEXT: CXXRecord
// CHECK-NEXT: ParmVarDecl
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: OpenACCRoutineDeclAttr
// CHECK-NEXT: seq clause
}

void usage() {
  DepS<int> s;
  TemplFunc<S>();
}

#endif
