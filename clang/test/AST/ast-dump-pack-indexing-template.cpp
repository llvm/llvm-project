// RUN: %clang_cc1 -std=c++2d -ast-dump -ast-dump-filter=Dump %s | FileCheck %s

template <class T> struct A {};
template <class T> struct B {};

template <template <class> class... TT>
struct DumpDependent {
  using type = TT...[1]<int>;
};

// CHECK-LABEL: Dumping DumpDependent:
// CHECK:      TypeAliasDecl {{.*}} type 'TT...[1]<int>'
// CHECK-NEXT: `-TemplateSpecializationType {{.*}} 'TT...[1]<int>' dependent
// CHECK-NEXT:   |-name: 'TT...[1]':'template-parameter-0-0...[1]' pack_indexing index 1
// CHECK-NEXT:   | |-pattern: 'TT':'template-parameter-0-0'
// CHECK-NEXT:   | | `-TemplateTemplateParmDecl {{.*}} depth 0 index 0 ... TT
// CHECK-NEXT:   | `-index: ConstantExpr {{.*}} '__size_t':'{{.*}}'
// CHECK-NEXT:   |   `-value: Int 1
// CHECK-NEXT:   `-TemplateArgument type 'int'

using DumpSubstituted = DumpDependent<A, B>::type;

// CHECK-LABEL: Dumping DumpSubstituted:
// CHECK:      TypeAliasDecl {{.*}} DumpSubstituted 'DumpDependent<A, B>::type':'B<int>'
// CHECK:      TemplateSpecializationType {{.*}} 'TT...[1]<int>' sugar
// CHECK-NEXT: |-name: 'TT...[1]':'B' pack_indexing fully_substituted index 1

template <class T> concept C = true;

template <template <class> concept... CC>
constexpr bool DumpConceptId = CC...[0]<int>;

// CHECK-LABEL: Dumping DumpConceptId:
// CHECK:      VarTemplateDecl {{.*}} DumpConceptId
// CHECK:      DependentTemplateIdExpr {{.*}} concept
// CHECK-NEXT: `-name: 'CC...[0]':'template-parameter-0-0...[0]' pack_indexing index 0
// CHECK-NEXT:   |-pattern: 'CC':'template-parameter-0-0'
// CHECK-NEXT:   | `-TemplateTemplateParmDecl {{.*}} depth 0 index 0 ... CC
// CHECK-NEXT:   `-index: ConstantExpr {{.*}} '__size_t':'{{.*}}'
// CHECK-NEXT:     `-value: Int 0

template <template <class> auto... VV>
constexpr int DumpVariableTemplateId = VV...[1]<int>;

// CHECK-LABEL: Dumping DumpVariableTemplateId:
// CHECK:      DependentTemplateIdExpr {{.*}} variable template
// CHECK-NEXT: `-name: 'VV...[1]':'template-parameter-0-0...[1]' pack_indexing index 1

template <template <class> concept... CC>
struct DumpTypeConstraint {
  template <CC...[0] T>
  static void f();
};

// CHECK-LABEL: Dumping DumpTypeConstraint:
// CHECK: TemplateTypeParmDecl {{.*}} Concept {{.*}} 'CC...[0]' depth 1 index 0 T


template <int N, template <class> class... TT>
struct DumpDependentIndex {
  using type = TT...[N + 1]<int>;
};
// CHECK-LABEL: Dumping DumpDependentIndex:
// CHECK:      TypeAliasDecl {{.*}} type 'TT...[N + 1]<int>'
// CHECK-NEXT: `-TemplateSpecializationType {{.*}} 'TT...[N + 1]<int>' dependent
// CHECK-NEXT:   |-name: 'TT...[N + 1]':'template-parameter-0-1...[N + 1]' pack_indexing
// CHECK-NEXT:   | |-pattern: 'TT':'template-parameter-0-1'
// CHECK-NEXT:   | | `-TemplateTemplateParmDecl {{.*}} depth 0 index 1 ... TT
// CHECK-NEXT:   | `-index: BinaryOperator {{.*}} 'int' '+'
// CHECK-NEXT:   `-TemplateArgument type 'int'
