// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump=json %s | FileCheck --check-prefix=JSON %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-print %s > %t
// RUN: FileCheck < %t %s -check-prefix=CHECK1
// RUN: FileCheck < %t %s -check-prefix=CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -ast-dump %s | FileCheck --check-prefix=DUMP %s

// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c++ -std=c++20 -include-pch %t \
// RUN: -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace --check-prefix=DUMP %s

template <int X, typename Y, int Z = 5>
struct foo {
  int constant;
  foo() {}
  Y getSum() { return Y(X + Z); }
};

template <int A, typename B>
B bar() {
  return B(A);
}

void baz() {
  int x = bar<5, int>();
  int y = foo<5, int>().getSum();
  double z = foo<2, double, 3>().getSum();
}

// Template definition - foo
// CHECK1: template <int X, typename Y, int Z = 5> struct foo {
// CHECK2: template <int X, typename Y, int Z = 5> struct foo {

// Template instantiation - foo
// Since the order of instantiation may vary during runs, run FileCheck twice
// to make sure each instantiation is in the correct spot.
// CHECK1: template<> struct foo<5, int, 5> {
// CHECK2: template<> struct foo<2, double, 3> {

// Template definition - bar
// CHECK1: template <int A, typename B> B bar()
// CHECK2: template <int A, typename B> B bar()

// Template instantiation - bar
// CHECK1: template<> int bar<5, int>()
// CHECK2: template<> int bar<5, int>()

// CHECK1-LABEL: template <typename ...T> struct A {
// CHECK1-NEXT:    template <T ...x[3]> struct B {
template <typename ...T> struct A {
  template <T ...x[3]> struct B {};
};

// CHECK1-LABEL: template <typename ...T> void f() {
// CHECK1-NEXT:    A<T[3]...> a;
template <typename ...T> void f() {
  A<T[3]...> a;
}

namespace test2 {
void func(int);
void func(float);
template<typename T>
void tmpl() {
  func(T());
}

// DUMP: UnresolvedLookupExpr {{.*}} <col:3> '<overloaded function type>' lvalue (ADL) = 'func'
}

namespace test3 {
  template<typename T> struct A {};
  template<typename T> A(T) -> A<int>;
  // CHECK1: template <typename T> A(T) -> A<int>;
}

namespace test4 {
template <unsigned X, auto A>
struct foo {
  static void fn();
};

// Prints using an "integral" template argument. Test that this correctly
// includes the type for the auto argument and omits it for the fixed
// type/unsigned argument (see
// TemplateParameterList::shouldIncludeTypeForArgument)
// CHECK1: {{^    }}template<> struct foo<0, 0L> {
// CHECK1: {{^    }}void test(){{ }}{
// CHECK1: {{^        }}foo<0, 0 + 0L>::fn();
void test() {
  foo<0, 0 + 0L>::fn();
}

// Prints using an "expression" template argument. This renders based on the way
// the user wrote the arguments (including that + expression) - so it's not
// powered by the shouldIncludeTypeForArgument functionality.
// Not sure if this it's intentional that these two specializations are rendered
// differently in this way.
// CHECK1: {{^    }}template<> struct foo<1, 0 + 0L> {
template struct foo<1, 0 + 0L>;
}

namespace test5 {
template<long> void f() {}
void (*p)() = f<0>;
template<unsigned = 0> void f() {}
void (*q)() = f<>;
// Not perfect - this code in the dump would be ambiguous, but it's the best we
// can do to differentiate these two implicit specializations.
// CHECK1: template<> void f<0L>()
// CHECK1: template<> void f<0U>()
}

namespace test6 {
template <class D>
constexpr bool C = true;

template <class Key>
void func() {
  C<Key>;
// DUMP:      UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (no ADL) = 'C'
// DUMP-NEXT: `-TemplateArgument type 'Key'
// DUMP-NEXT:   `-TemplateTypeParmType {{.*}} 'Key' dependent depth 0 index 0
// DUMP-NEXT:     `-TemplateTypeParm {{.*}} 'Key'
}
}

namespace test7 {
  template <template<class> class TT> struct AA {};
  template <class...> class B {};
  template struct AA<B>;
// DUMP-LABEL: NamespaceDecl {{.*}} test7{{$}}
// DUMP:       ClassTemplateDecl 0x{{.+}} AA{{$}}
// DUMP-NEXT:  |-TemplateTemplateParmDecl
// DUMP-NEXT:  | `-TemplateTypeParmDecl
// DUMP-NEXT:  |-CXXRecordDecl 0x[[TEST7_PAT:[^ ]+]] {{.+}} struct AA definition
// DUMP:       ClassTemplateSpecializationDecl {{.*}} struct AA definition instantiated_from 0x[[TEST7_PAT]] explicit_instantiation_definition strict-pack-match{{$}}

// JSON:       "name": "test7",
// JSON:        "kind": "ClassTemplateSpecializationDecl",
// JSON:        "name": "AA",
// JSON-NEXT:   "tagUsed": "struct",
// JSON-NEXT:   "completeDefinition": true,
// JSON-NEXT:   "strict-pack-match": true,
// JSON:       "name": "test8",
} // namespce test7

namespace test8 {
template<_Complex int x>
struct pr126341;
template<>
struct pr126341<{1, 2}>;
// DUMP-LABEL: NamespaceDecl {{.*}} test8{{$}}
// DUMP-NEXT:  |-ClassTemplateDecl {{.*}} pr126341
// DUMP:       `-ClassTemplateSpecializationDecl {{.*}} pr126341
// DUMP:         `-TemplateArgument structural value '1+2i'
} // namespace test8

namespace TestMemberPointerPartialSpec {
  template <class> struct A;
  template <class T1, class T2> struct A<T1 T2::*>;
// DUMP-LABEL: NamespaceDecl {{.+}} TestMemberPointerPartialSpec{{$}}
// DUMP:       ClassTemplatePartialSpecializationDecl {{.*}} struct A
// DUMP-NEXT:  |-TemplateArgument type 'type-parameter-0-0 type-parameter-0-1::*'
// DUMP-NEXT:  | `-MemberPointerType {{.+}} 'type-parameter-0-0 type-parameter-0-1::*' dependent
// DUMP-NEXT:  |   |-TemplateTypeParmType {{.+}} 'type-parameter-0-1' dependent depth 0 index 1
// DUMP-NEXT:  |   `-TemplateTypeParmType {{.+}} 'type-parameter-0-0' dependent depth 0 index 0
} // namespace TestMemberPointerPartialSpec

namespace TestDependentMemberPointer {
  template <class U> struct A {
    using X = int U::*;
    using Y = int U::test::*;
    using Z = int U::template V<int>::*;
  };
// DUMP-LABEL: NamespaceDecl {{.+}} TestDependentMemberPointer{{$}}
// DUMP:       |-TypeAliasDecl {{.+}} X 'int U::*'{{$}}
// DUMP-NEXT:  | `-MemberPointerType {{.+}} 'int U::*' dependent
// DUMP-NEXT:  |   |-TemplateTypeParmType {{.+}} 'U' dependent depth 0 index 0
// DUMP-NEXT:  |   | `-TemplateTypeParm {{.+}} 'U'
// DUMP-NEXT:  |   `-BuiltinType {{.+}} 'int'
// DUMP-NEXT:  |-TypeAliasDecl {{.+}} Y 'int U::test::*'{{$}}
// DUMP-NEXT:  | `-MemberPointerType {{.+}} 'int U::test::*' dependent
// DUMP-NEXT:  |   |-DependentNameType {{.+}} 'U::test' dependent
// DUMP-NEXT:  |   `-BuiltinType {{.+}} 'int'
// DUMP-NEXT:  `-TypeAliasDecl {{.+}} Z 'int U::template V<int>::*'{{$}}
// DUMP-NEXT:    `-MemberPointerType {{.+}} 'int U::template V<int>::*' dependent
// DUMP-NEXT:      |-TemplateSpecializationType {{.+}} 'U::template V<int>' dependent
// DUMP-NEXT:      | |-name: 'U::template V':'type-parameter-0-0::template V' dependent
// DUMP-NEXT:      | | `-NestedNameSpecifier TypeSpec 'U'
// DUMP-NEXT:      | `-TemplateArgument type 'int'
// DUMP-NEXT:      `-BuiltinType {{.+}} 'int'
} // namespace TestDependentMemberPointer

namespace TestPartialSpecNTTP {
// DUMP-LABEL: NamespaceDecl {{.+}} TestPartialSpecNTTP{{$}}
  template <class TA1, bool TA2> struct Template1 {};
  template <class TB1, bool TB2> struct Template2 {};

  template <class U1, bool U2, bool U3>
  struct Template2<Template1<U1, U2>, U3> {};
// DUMP:      ClassTemplatePartialSpecializationDecl {{.+}} struct Template2
// DUMP:      |-TemplateArgument type 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>'
// DUMP-NEXT: | `-TemplateSpecializationType {{.+}} 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-1>' dependent
// DUMP-NEXT: |   |-name: 'TestPartialSpecNTTP::Template1'
// DUMP-NEXT: |   | `-ClassTemplateDecl {{.+}} Template1
// DUMP-NEXT: |   |-TemplateArgument type 'type-parameter-0-0'
// DUMP-NEXT: |   | `-TemplateTypeParmType {{.+}} 'type-parameter-0-0' dependent depth 0 index 0
// DUMP-NEXT: |   `-TemplateArgument expr canonical 'value-parameter-0-1'
// DUMP-NEXT: |     `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U2' 'bool'
// DUMP-NEXT: |-TemplateArgument expr canonical 'value-parameter-0-2'
// DUMP-NEXT: | `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U3' 'bool'
// DUMP-NEXT: |-TemplateTypeParmDecl {{.+}} referenced class depth 0 index 0 U1
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 1 U2
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 2 U3
// DUMP-NEXT: `-CXXRecordDecl {{.+}} implicit struct Template2

  template <typename U1, bool U3, bool U2>
  struct Template2<Template1<U1, U2>, U3> {};
// DUMP:      ClassTemplatePartialSpecializationDecl {{.+}} struct Template2 definition explicit_specialization
// DUMP:      |-TemplateArgument type 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>'
// DUMP-NEXT: | `-TemplateSpecializationType {{.+}} 'TestPartialSpecNTTP::Template1<type-parameter-0-0, value-parameter-0-2>' dependent
// DUMP-NEXT: |   |-name: 'TestPartialSpecNTTP::Template1'
// DUMP-NEXT: |   | `-ClassTemplateDecl {{.+}} Template1
// DUMP-NEXT: |   |-TemplateArgument type 'type-parameter-0-0'
// DUMP-NEXT: |   | `-TemplateTypeParmType {{.+}} 'type-parameter-0-0' dependent depth 0 index 0
// DUMP-NEXT: |   `-TemplateArgument expr canonical 'value-parameter-0-2'
// DUMP-NEXT: |     `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U2' 'bool'
// DUMP-NEXT: |-TemplateArgument expr canonical 'value-parameter-0-1'
// DUMP-NEXT: | `-DeclRefExpr {{.+}} 'bool' NonTypeTemplateParm {{.+}} 'U3' 'bool'
// DUMP-NEXT: |-TemplateTypeParmDecl {{.+}} referenced typename depth 0 index 0 U1
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 1 U3
// DUMP-NEXT: |-NonTypeTemplateParmDecl {{.+}} referenced 'bool' depth 0 index 2 U2
// DUMP-NEXT: `-CXXRecordDecl {{.+}} implicit struct Template2
} // namespace TestPartialSpecNTTP

namespace GH153540 {
// DUMP-LABEL: NamespaceDecl {{.*}} GH153540{{$}}

  namespace N {
    template<typename T> struct S { S(T); };
  }
  void f() {
    N::S(0);
  }

// DUMP:      FunctionDecl {{.*}} f 'void ()'
// DUMP-NEXT: CompoundStmt
// DUMP-NEXT: CXXFunctionalCastExpr {{.*}} 'N::S<int>':'GH153540::N::S<int>'
// DUMP-NEXT: CXXConstructExpr {{.*}} <col:5, col:11> 'N::S<int>':'GH153540::N::S<int>' 'void (int)'
} // namespace GH153540

namespace AliasDependentTemplateSpecializationType {
  // DUMP-LABEL: NamespaceDecl {{.*}} AliasDependentTemplateSpecializationType{{$}}

  template<template<class> class TT> using T1 = TT<int>;
  template<class T> using T2 = T1<T::template X>;

// DUMP:      TypeAliasDecl {{.*}} T2 'T1<T::template X>':'T::template X<int>'
// DUMP-NEXT: `-TemplateSpecializationType {{.*}} 'T1<T::template X>' sugar dependent alias
// DUMP-NEXT:   |-name: 'T1':'AliasDependentTemplateSpecializationType::T1' qualified
// DUMP-NEXT:   | `-TypeAliasTemplateDecl {{.*}} T1
// DUMP-NEXT:   |-TemplateArgument template 'T::template X':'type-parameter-0-0::template X' dependent
// DUMP-NEXT:   | `-NestedNameSpecifier TypeSpec 'T'
// DUMP-NEXT:   `-TemplateSpecializationType {{.*}} 'T::template X<int>' dependent
// DUMP-NEXT:     |-name: 'T::template X':'type-parameter-0-0::template X' subst index 0 final
// DUMP-NEXT:     | |-parameter: TemplateTemplateParmDecl {{.*}} depth 0 index 0 TT
// DUMP-NEXT:     | |-associated TypeAliasTemplate {{.*}} 'T1'
// DUMP-NEXT:     | `-replacement: 'T::template X':'type-parameter-0-0::template X' dependent
// DUMP-NEXT:     |   `-NestedNameSpecifier TypeSpec 'T'
// DUMP-NEXT:     `-TemplateArgument type 'int'
// DUMP-NEXT:       `-BuiltinType {{.*}} 'int'
} // namespace

namespace TestAbbreviatedTemplateDecls {
  // DUMP-LABEL: NamespaceDecl {{.*}} TestAbbreviatedTemplateDecls{{$}}
  void abbreviated(auto);
  template<class T>
  void mixed(T, auto);

// DUMP: FunctionTemplateDecl {{.*}} <line:[[@LINE-4]]:3, col:24> col:8 abbreviated
// DUMP: FunctionTemplateDecl {{.*}} <line:[[@LINE-4]]:3, line:[[@LINE-3]]:21> col:8 mixed

} // namespace TestAbbreviatedTemplateDecls
