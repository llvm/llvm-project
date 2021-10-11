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
}
using ns::S;

// TemplateName in TemplateSpecializationType.
template<typename T>
using A = S<T>;
// CHECK:      TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType {{.*}} 'S<T>' sugar dependent
// CHECK-NEXT:   `-TemplateSpecializationType {{.*}} 'S<T>' dependent using S

// TemplateName in TemplateArgument.
template <template <typename> class T> class X {};
using B = X<S>;
// CHECK:      TypeAliasDecl
// CHECK-NEXT: `-ElaboratedType {{.*}} 'X<ns::S>' sugar
// CHECK-NEXT:   `-TemplateSpecializationType {{.*}} 'X<ns::S>' sugar X
// CHECK-NEXT:     |-TemplateArgument using template S
// CHECK-NEXT:       `-RecordType {{.*}} 'X<ns::S>'
// CHECK-NEXT:         `-ClassTemplateSpecialization {{.*}} 'X'

// TemplateName in DeducedTemplateSpecializationType.
S DeducedTemplateSpecializationT(123);
using C = decltype(DeducedTemplateSpecializationT);
// CHECK:      DecltypeType {{.*}}
// CHECK-NEXT:  |-DeclRefExpr {{.*}}
// CHECK-NEXT:  `-ElaboratedType {{.*}} 'S<int>' sugar
// CHECK-NEXT:    `-DeducedTemplateSpecializationType {{.*}} 'ns::S<int>' sugar using
