// RUN: %clang_cc1 -std=c++26 -ast-dump -ast-dump-filter=Test %s | FileCheck %s

template <template <class> class TT> using N = TT<int>;

namespace qualified {
  namespace foo {
    template <class T> struct A;
  } // namespace foo
  using TestQualified = N<foo::A>;
} // namespace qualified

// CHECK:      Dumping qualified::TestQualified:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TemplateSpecializationType
// CHECK-NEXT:     |-name: 'N' qualified
// CHECK-NEXT:     | `-TypeAliasTemplateDecl {{.+}} N{{$}}
// CHECK-NEXT:     |-TemplateArgument template 'foo::A':'qualified::foo::A' qualified{{$}}
// CHECK-NEXT:     | |-NestedNameSpecifier Namespace 0x{{.+}} 'foo'{{$}}
// CHECK-NEXT:     | `-ClassTemplateDecl {{.+}} A{{$}}

namespace dependent {
  template <class T> struct B {
    using TestDependent = N<T::template X>;
  };
} // namespace dependent

// CHECK:      Dumping dependent::B::TestDependent:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TemplateSpecializationType
// CHECK-NEXT:     |-name: 'N' qualified
// CHECK-NEXT:     | `-TypeAliasTemplateDecl
// CHECK-NEXT:     |-TemplateArgument template 'T::template X':'type-parameter-0-0::template X' dependent{{$}}
// CHECK-NEXT:     | `-NestedNameSpecifier TypeSpec 'T'{{$}}

namespace subst {
  template <class> struct A;

  template <template <class> class TT> struct B {
    template <template <class> class> struct C {};
    using type = C<TT>;
  };
  using TestSubst = B<A>::type;
} // namespace subst

// CHECK:      Dumping subst::TestSubst:
// CHECK-NEXT: TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType
// CHECK-NEXT:   `-TypedefType
// CHECK-NEXT:     |-TypeAlias
// CHECK-NEXT:     `-ElaboratedType
// CHECK-NEXT:       `-TemplateSpecializationType
// CHECK-NEXT:         |-name: 'C':'subst::B<subst::A>::C' qualified
// CHECK-NEXT:         | `-ClassTemplateDecl {{.+}} C
// CHECK-NEXT:         |-TemplateArgument template 'subst::A' subst index 0
// CHECK-NEXT:         | |-parameter: TemplateTemplateParmDecl {{.+}} depth 0 index 0 TT{{$}}
// CHECK-NEXT:         | |-associated ClassTemplateSpecialization {{.+}} 'B'{{$}}
// CHECK-NEXT:         | `-replacement:
// CHECK-NEXT:         |   `-ClassTemplateDecl {{.+}} A{{$}}
