// An unknown attribute on a template member survives instantiation: the cloned
// attribute on the specialization carries the interned argument text. Because
// UnknownAttr stores the argument text (not a source range), it can be created
// implicitly for the instantiation, which has no source of its own.

// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -ast-dump %s | FileCheck %s

template <class T> struct S {
  T x [[ns::transient(a, b)]];
};
template struct S<int>;

// CHECK: ClassTemplateSpecializationDecl {{.*}} struct S definition
// CHECK: FieldDecl {{.*}} x 'int'
// CHECK-NEXT: UnknownAttr {{.*}} ns::transient "(a, b)"

// A dependent argument is retained verbatim. The attribute is ignored, so its
// arguments are never parsed, substituted, or checked: the specialization keeps
// the same text as the template, and instantiating with a type that has no such
// member is not an error.
template <class T> struct D {
  int y [[vendor::attr(T::value + 1)]];
};
struct NoValue {};
template struct D<NoValue>;

// CHECK: ClassTemplateSpecializationDecl {{.*}} struct D definition
// CHECK: FieldDecl {{.*}} y 'int'
// CHECK-NEXT: UnknownAttr {{.*}} vendor::attr "(T::value + 1)"
