// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++2a -ast-dump %s | FileCheck -strict-whitespace %s

template <typename, typename>
constexpr bool Concept = true;
template<typename T> // depth 0
struct Out {
  template<typename U> // depth 1
  struct Inner {
    U t;
  };

  template<typename V> // depth1
  requires Concept<T, V>
  Inner(V) -> Inner<V>;
};

template <typename X>
struct Out2 {
  template<typename Y> // depth1
  using AInner = Out<int>::Inner<Y>;
};
Out2<double>::AInner t(1.0);

// Verify that the require-clause of alias deduction guide is transformed correctly:
//   - Occurrence T should be replaced with `int`;
//   - Occurrence V should be replaced with the Y with depth 1
//   - Depth of occurrence Y in the __is_deducible constraint should be 1
//
// CHECK:      |   `-FunctionTemplateDecl {{.*}} <deduction guide for AInner>
// CHECK-NEXT: |     |-TemplateTypeParmDecl {{.*}} typename depth 0 index 0 Y
// CHECK-NEXT: |     |-BinaryOperator {{.*}} '<dependent type>' '&&'
// CHECK-NEXT: |     | |-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (no ADL) = 'Concept'
// CHECK-NEXT: |     | | |-TemplateArgument type 'int'
// CHECK-NEXT: |     | | | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT: |     | | `-TemplateArgument type 'Y':'type-parameter-1-0'
// CHECK-NEXT: |     | |   `-TemplateTypeParmType {{.*}} 'Y' dependent depth 1 index 0
// CHECK-NEXT: |     | |     `-TemplateTypeParm {{.*}} 'Y'
// CHECK-NEXT: |     | `-TypeTraitExpr {{.*}} 'bool' __is_deducible
// CHECK-NEXT: |     |   |-DeducedTemplateSpecializationType {{.*}} 'Out2<double>::AInner' dependent
// CHECK-NEXT: |     |   | `-name: 'Out2<double>::AInner'
// CHECK-NEXT: |     |   |   `-TypeAliasTemplateDecl {{.+}} AInner{{$}}
// CHECK-NEXT: |     |   `-TemplateSpecializationType {{.*}} 'Inner<Y>' dependent
// CHECK-NEXT: |     |     |-name: 'Inner':'Out<int>::Inner' qualified
// CHECK-NEXT: |     |     | `-ClassTemplateDecl {{.+}} Inner{{$}}
// CHECK-NEXT: |     |     `-TemplateArgument type 'Y'
// CHECK-NEXT: |     |       `-SubstTemplateTypeParmType {{.*}} 'Y'
// CHECK-NEXT: |     |         |-FunctionTemplate {{.*}} '<deduction guide for Inner>'
// CHECK-NEXT: |     |         `-TemplateTypeParmType {{.*}} 'Y' dependent depth 1 index 0
// CHECK-NEXT: |     |           `-TemplateTypeParm {{.*}} 'Y'
// CHECK-NEXT: |     |-CXXDeductionGuideDecl {{.*}} <deduction guide for AInner> 'auto (Y) -> Inner<Y>'
// CHECK-NEXT: |     | `-ParmVarDecl {{.*}} 'Y'
// CHECK-NEXT: |     `-CXXDeductionGuideDecl {{.*}} used <deduction guide for AInner> 'auto (double) -> Inner<double>' implicit_instantiation
// CHECK-NEXT: |       |-TemplateArgument type 'double'
// CHECK-NEXT: |       | `-BuiltinType {{.*}} 'double'
// CHECK-NEXT: |       `-ParmVarDecl {{.*}} 'double'

// GH92596
template <typename T0>
struct Out3 {
  template<class T1, typename T2>
  struct Foo {
    // Deduction guide:
    //   template <typename T1, typename T2, typename V>
    //   Foo(V, T1) -> Foo<T1, T2>;
    template<class V> requires Concept<T0, V> // V in require clause of Foo deduction guide: depth 1, index: 2
    Foo(V, T1);
  };
};
template<class T3>
using AFoo3 = Out3<int>::Foo<T3, T3>;
AFoo3 afoo3{0, 1};
// Verify occurrence V in the require-clause is transformed (depth: 1 => 0, index: 2 => 1) correctly.

// CHECK:      FunctionTemplateDecl {{.*}} implicit <deduction guide for AFoo3>
// CHECK-NEXT: |-TemplateTypeParmDecl {{.*}} class depth 0 index 0 T3
// CHECK-NEXT: |-TemplateTypeParmDecl {{.*}} class depth 0 index 1 V
// CHECK-NEXT: |-BinaryOperator {{.*}} '<dependent type>' '&&'
// CHECK-NEXT: | |-UnresolvedLookupExpr {{.*}} '<dependent type>' lvalue (no ADL) = 'Concept'
// CHECK-NEXT: | | |-TemplateArgument type 'int'
// CHECK-NEXT: | | | `-BuiltinType {{.*}} 'int'
// CHECK-NEXT: | | `-TemplateArgument type 'V'
// CHECK-NEXT: | |   `-TemplateTypeParmType {{.*}} 'V' dependent depth 0 index 1

template <typename... T1>
struct Foo {
  Foo(T1...);
};

template <typename...T2>
using AFoo = Foo<T2...>;
AFoo a(1, 2);
// CHECK:      |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for AFoo> 'auto (T2...) -> Foo<T2...>'
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'T2...' pack
// CHECK-NEXT: | `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for AFoo> 'auto (int, int) -> Foo<int, int>' implicit_instantiation

template <typename T>
using BFoo = Foo<T, T>;
BFoo b2(1.0, 2.0);
// CHECK:      |-CXXDeductionGuideDecl {{.*}} implicit <deduction guide for BFoo> 'auto (T, T) -> Foo<T, T>'
// CHECK-NEXT: | | |-ParmVarDecl {{.*}} 'T'
// CHECK-NEXT: | | `-ParmVarDecl {{.*}} 'T'
// CHECK-NEXT: | `-CXXDeductionGuideDecl {{.*}} implicit used <deduction guide for BFoo> 'auto (double, double) -> Foo<double, double>' implicit_instantiation

namespace GH90209 {
// Case 1: type template parameter
template <class Ts>
struct List1 {
  List1(int);
};

template <class T1>
struct TemplatedClass1 {
  TemplatedClass1(T1);
};

template <class T1>
TemplatedClass1(T1) -> TemplatedClass1<List1<T1>>;

template <class T2>
using ATemplatedClass1 = TemplatedClass1<List1<T2>>;

ATemplatedClass1 test1(1);
// Verify that we have a correct template parameter list for the deduction guide.
//
// CHECK:      FunctionTemplateDecl {{.*}} <deduction guide for ATemplatedClass1>
// CHECK-NEXT: |-TemplateTypeParmDecl {{.*}} class depth 0 index 0 T2
// CHECK-NEXT: |-TypeTraitExpr {{.*}} 'bool' __is_deducible

// Case 2: template template parameter
template<typename K> struct Foo{};

template <template<typename> typename Ts>
struct List2 {
  List2(int);
};

template <typename T1>
struct TemplatedClass2 {
  TemplatedClass2(T1);
};

template <template<typename> typename T1>
TemplatedClass2(T1<int>) -> TemplatedClass2<List2<T1>>;

template <template<typename> typename T2>
using ATemplatedClass2 = TemplatedClass2<List2<T2>>;

List2<Foo> list(1);
ATemplatedClass2 test2(list);
// Verify that we have a correct template parameter list for the deduction guide.
//
// CHECK:      FunctionTemplateDecl {{.*}} <deduction guide for ATemplatedClass2>
// CHECK-NEXT: |-TemplateTemplateParmDecl {{.*}} depth 0 index 0 T2
// CHECK-NEXT: | `-TemplateTypeParmDecl {{.*}} typename depth 0 index 0
// CHECK-NEXT: |-TypeTraitExpr {{.*}} 'bool' __is_deducible

} // namespace GH90209

namespace GH124715 {

template <class T, class... Args>
concept invocable = true;

template <class T, class... Args> struct Struct {
  template <class U>
    requires invocable<U, Args...>
  Struct(U, Args...) {}
};

template <class...> struct Packs {};

template <class Lambda, class... Args>
Struct(Lambda lambda, Args... args) -> Struct<Lambda, Args...>;

template <class T, class... Ts> using Alias = Struct<T, Packs<Ts...>>;

void foo() {
  Alias([](int) {}, Packs<int>());
}

// CHECK:      |-FunctionTemplateDecl {{.*}} implicit <deduction guide for Alias>
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} class depth 0 index 0 T
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} class depth 0 index 1 ... Ts
// CHECK-NEXT: | |-TemplateTypeParmDecl {{.*}} class depth 0 index 2 U
// CHECK-NEXT: | |-BinaryOperator {{.*}} 'bool' '&&'
// CHECK-NEXT: | | |-ConceptSpecializationExpr {{.*}} 'bool' Concept {{.*}} 'invocable'
// CHECK-NEXT: | | | |-ImplicitConceptSpecializationDecl {{.*}}
// CHECK-NEXT: | | | | |-TemplateArgument type 'U'
// CHECK-NEXT: | | | | | `-TemplateTypeParmType {{.*}} 'U' dependent depth 0 index 2
// CHECK-NEXT: | | | | |   `-TemplateTypeParm {{.*}} 'U'
// CHECK-NEXT: | | | | `-TemplateArgument pack '<Packs<Ts...>>'
// CHECK-NEXT: | | | |   `-TemplateArgument type 'Packs<Ts...>'
// CHECK-NEXT: | | | |     `-TemplateSpecializationType {{.*}} 'Packs<Ts...>' dependent
// CHECK-NEXT: | | | |       |-name: 'Packs':'GH124715::Packs' qualified
// CHECK-NEXT: | | | |       | `-ClassTemplateDecl {{.*}} Packs
// CHECK-NEXT: | | | |       `-TemplateArgument type 'Ts...'
// CHECK-NEXT: | | | |         `-PackExpansionType {{.*}} 'Ts...' dependent
// CHECK-NEXT: | | | |           `-TemplateTypeParmType {{.*}} 'Ts' dependent contains_unexpanded_pack depth 0 index 1 pack
// CHECK-NEXT: | | | |             `-TemplateTypeParm {{.*}} 'Ts'
// CHECK-NEXT: | | | |-TemplateArgument {{.*}} type 'U':'type-parameter-0-2'
// CHECK-NEXT: | | | | `-TemplateTypeParmType {{.*}} 'U' dependent depth 0 index 2
// CHECK-NEXT: | | | |   `-TemplateTypeParm {{.*}} 'U'

} // namespace GH124715
