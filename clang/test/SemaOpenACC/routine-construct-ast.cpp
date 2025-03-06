// RUN: %clang_cc1 %s -fopenacc -ast-dump | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -fopenacc -emit-pch -o %t %s
// RUN: %clang_cc1 %s -fopenacc -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER
auto Lambda = [](){};
#pragma acc routine(Lambda)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' '(lambda at
int function();
#pragma acc routine (function)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'function' 'int ()'

namespace NS {
  // CHECK-NEXT: NamespaceDecl
  int NSFunc();
auto Lambda = [](){};
}
#pragma acc routine(NS::NSFunc)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'NSFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'NS'
#pragma acc routine(NS::Lambda)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'NS::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'NS'

struct S {
  void MemFunc();
  static void StaticMemFunc();
  constexpr static auto Lambda = [](){};
#pragma acc routine(S::MemFunc)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
#pragma acc routine(S::StaticMemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
#pragma acc routine(S::Lambda)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'

#pragma acc routine(MemFunc)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
#pragma acc routine(StaticMemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
#pragma acc routine(Lambda)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
};

#pragma acc routine(S::MemFunc)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
#pragma acc routine(S::StaticMemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
#pragma acc routine(S::Lambda)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'

template<typename T>
struct DepS {
  T MemFunc();
  static T StaticMemFunc();
  constexpr static auto Lambda = [](){};

#pragma acc routine(Lambda)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const auto'
#pragma acc routine(MemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'T ()'
#pragma acc routine(StaticMemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'T ()'

#pragma acc routine(DepS::Lambda)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const auto'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
#pragma acc routine(DepS::MemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
#pragma acc routine(DepS::StaticMemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'

#pragma acc routine(DepS<T>::Lambda)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const auto'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
#pragma acc routine(DepS<T>::MemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'
#pragma acc routine(DepS<T>::StaticMemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'T ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<T>'

// Instantiation:
// CHECK: ClassTemplateSpecializationDecl{{.*}}struct DepS
// CHECK: CXXRecordDecl{{.*}} struct DepS

// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'

// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
};

#pragma acc routine(DepS<int>::Lambda)
// CHECK: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const DepS<int>::
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
#pragma acc routine(DepS<int>::MemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'
#pragma acc routine(DepS<int>::StaticMemFunc)
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'int ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'DepS<int>'

template<typename T>
void TemplFunc() {
#pragma acc routine(T::MemFunc)
// CHECK: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
#pragma acc routine(T::StaticMemFunc)
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'
#pragma acc routine(T::Lambda)
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DependentScopeDeclRefExpr{{.*}}'<dependent type>'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'T'

// Instantiation:
// CHECK: FunctionDecl{{.*}} TemplFunc 'void ()' implicit_instantiation
// CHECK: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'MemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'StaticMemFunc' 'void ()'
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: OpenACCRoutineDecl{{.*}} routine name_specified
// CHECK-NEXT: DeclRefExpr{{.*}} 'Lambda' 'const S::(lambda at
// CHECK-NEXT: NestedNameSpecifier{{.*}} 'S'
}

void usage() {
  DepS<int> s;
  TemplFunc<S>();
}
#endif
