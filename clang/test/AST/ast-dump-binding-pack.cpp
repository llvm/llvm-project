// RUN: %clang_cc1 -ast-dump -std=c++26 %s | FileCheck %s

// Test this with PCH.
// RUN: %clang_cc1 %s -std=c++26 -emit-pch -o %t %s
// RUN: %clang_cc1 %s -std=c++26 -include-pch %t -ast-dump-all | FileCheck %s

#ifndef PCH_HELPER
#define PCH_HELPER

template <unsigned N>
void foo() {
  int arr[4] = {1, 2, 3, 4};
  auto [binding_1, ...binding_rest, binding_4] = arr;
  int arr_2[] = {binding_rest...};
};

// CHECK-LABEL: FunctionTemplateDecl {{.*}} foo
// CHECK-LABEL: BindingDecl {{.*}} binding_1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}
// CHECK-NEXT: ImplicitCastExpr
// CHECK-NEXT: DeclRefExpr {{.*}}
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NOT: BindingDecl
// CHECK-LABEL: BindingDecl {{.*}} binding_rest
// CHECK-NEXT: ResolvedUnexpandedPackExpr
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue Binding {{.*}} 'binding_rest'
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue Binding {{.*}} 'binding_rest'
// CHECK-NOT: BindingDecl
// CHECK-LABEL: BindingDecl {{.*}} binding_4
// CHECK-NEXT: ArraySubscriptExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue Decomposition {{.*}}
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 3
// CHECK-NOT: BindingDecl
// CHECK-LABEL: VarDecl {{.*}} arr_2
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: PackExpansionExpr {{.*}} '<dependent type>' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} lvalue Binding {{.*}} 'binding_rest'

struct tag_t { };
template <unsigned N>
void bar() {
  auto [...empty_binding_pack] = tag_t{};
  static_assert(sizeof...(empty_binding_pack) == 0);
};

// CHECK-LABEL: FunctionTemplateDecl {{.*}} bar
// CHECK-NOT: BindingDecl
// CHECK-LABEL: BindingDecl {{.*}} empty_binding_pack
// CHECK-NEXT: ResolvedUnexpandedPackExpr
// CHECK-NOT: DeclRefExpr {{.*}} 'empty_binding_pack'
// CHECK-NOT: BindingDecl
// CHECK: DeclStmt

struct int_pair { int x; int y; };
template <typename T>
void baz() {
  auto [binding_1, binding_2, ...empty_binding_pack] = T{};
  static_assert(sizeof...(empty_binding_pack) == 0);
};

void(*f)() = baz<int_pair>;

// CHECK-LABEL: FunctionDecl {{.*}} baz {{.*}} implicit_instantiation
// CHECK-NEXT: TemplateArgument type 'int_pair'
// CHECK: BindingDecl {{.*}} binding_1
// CHECK: BindingDecl {{.*}} binding_2
// CHECK-NOT: BindingDecl
// CHECK-LABEL: BindingDecl {{.*}} empty_binding_pack
// CHECK-NEXT: ResolvedUnexpandedPackExpr
// CHECK-NOT: DeclRefExpr {{.*}} 'empty_binding_pack'
// CHECK-NOT: BindingDecl
// CHECK: DeclStmt
#endif
