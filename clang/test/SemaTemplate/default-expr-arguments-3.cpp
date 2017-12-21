// RUN: %clang_cc1 -std=c++14 -verify -ast-dump %s | FileCheck %s
// expected-no-diagnostics

// CHECK: FunctionDecl {{.*}} used func 'void (void)'
// CHECK-NEXT: TemplateArgument type 'int'
// CHECK: LambdaExpr {{.*}} 'class (lambda at
// CHECK: ParmVarDecl {{.*}} used f 'enum foo' cinit
// CHECK-NEXT: DeclRefExpr {{.*}} 'enum foo' EnumConstant {{.*}} 'a' 'enum foo'

namespace PR28795 {
  template<typename T>
  void func() {
    enum class foo { a, b };
    auto bar = [](foo f = foo::a) { return f; };
    bar();
  }

  void foo() {
    func<int>();
  }
}

// CHECK: ClassTemplateSpecializationDecl {{.*}} struct class2 definition
// CHECK: TemplateArgument type 'int'
// CHECK: LambdaExpr {{.*}} 'class (lambda at
// CHECK: ParmVarDecl {{.*}} used f 'enum foo' cinit
// CHECK-NEXT: DeclRefExpr {{.*}} 'enum foo' EnumConstant {{.*}} 'a' 'enum foo'

// Template struct case:
template <class T> struct class2 {
  void bar() {
    enum class foo { a, b };
    [](foo f = foo::a) { return f; }();
  }
};

template struct class2<int>;

// CHECK: FunctionTemplateDecl {{.*}} f1
// CHECK-NEXT: TemplateTypeParmDecl {{.*}} typename depth 0 index 0 T
// CHECK-NEXT: FunctionDecl {{.*}} f1 'void (void)'
// CHECK: FunctionDecl {{.*}} f1 'void (void)'
// CHECK-NEXT: TemplateArgument type 'int'
// CHECK: ParmVarDecl {{.*}} n 'enum foo' cinit
// CHECK-NEXT: DeclRefExpr {{.*}} 'enum foo' EnumConstant {{.*}} 'a' 'enum foo'

template<typename T>
void f1() {
  enum class foo { a, b };
  struct S {
    int g1(foo n = foo::a);
  };
}

template void f1<int>();
