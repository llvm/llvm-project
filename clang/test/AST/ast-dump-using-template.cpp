// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++17 -ast-dump %s | FileCheck -strict-whitespace %s

// Tests to verify we construct correct using template names.
// TemplateNames are not dumped, so the sugar here isn't obvious. However
// the "using" on the TemplateSpecializationTypes shows that the
// UsingTemplateName is present.
namespace ns {
template<typename T> class S {
 public:
   S(T);
};
template<typename T> struct S2 { S2(T); };
template <typename T> S2(T t) -> S2<T>;
}
using ns::S;
using ns::S2;

// TemplateName in TemplateSpecializationType.
template<typename T>
using A = S<T>;
// CHECK:      TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType {{.*}} 'S<T>' sugar dependent
// CHECK-NEXT:   `-TemplateSpecializationType {{.*}} 'S<T>' dependent
// CHECK-NEXT:     |-name: 'S':'ns::S' qualified
// CHECK-NEXT:     | |-UsingShadowDecl {{.+}} ClassTemplate {{.+}} 'S'

// TemplateName in TemplateArgument.
template <template <typename> class T> class X {};
using B = X<S>;
// CHECK:      TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType {{.*}} 'X<S>' sugar
// CHECK-NEXT:   `-TemplateSpecializationType {{.*}} 'X<S>' sugar
// CHECK-NEXT:     |-name: 'X' qualified
// CHECK-NEXT:     | `-ClassTemplateDecl {{.+}} X
// CHECK-NEXT:     |-TemplateArgument template 'S':'ns::S' qualified
// CHECK-NEXT:     | |-UsingShadowDecl {{.*}} implicit ClassTemplate {{.*}} 'S'
// CHECK-NEXT:     | `-target: ClassTemplateDecl {{.*}} S
// CHECK-NEXT:     `-RecordType {{.*}} 'X<ns::S>'
// CHECK-NEXT:       `-ClassTemplateSpecialization {{.*}} 'X'

// TemplateName in DeducedTemplateSpecializationType.
S DeducedTemplateSpecializationT(123);
using C = decltype(DeducedTemplateSpecializationT);
// CHECK:      DecltypeType {{.*}}
// CHECK-NEXT:  |-DeclRefExpr {{.*}}
// CHECK-NEXT:  `-ElaboratedType {{.*}} 'S<int>' sugar
// CHECK-NEXT:    `-DeducedTemplateSpecializationType {{.*}} 'ns::S<int>' sugar
// CHECK-NEXT:      |-name: 'S':'ns::S' qualified
// CHECK-NEXT:      | |-UsingShadowDecl {{.+}} 'S'

S2 DeducedTemplateSpecializationT2(123);
using D = decltype(DeducedTemplateSpecializationT2);
// CHECK:      DecltypeType {{.*}}
// CHECK-NEXT:  |-DeclRefExpr {{.*}}
// CHECK-NEXT:  `-ElaboratedType {{.*}} 'S2<int>' sugar
// CHECK-NEXT:    `-DeducedTemplateSpecializationType {{.*}} 'S2<int>' sugar
// CHECK-NEXT:      |-name: 'S2':'ns::S2' qualified
// CHECK-NEXT:       | |-UsingShadowDecl {{.+}} ClassTemplate {{.+}} 'S2'
