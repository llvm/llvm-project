// An unknown [[...]] attribute that appertains to a type is retained on an
// AttributedType as an UnknownTypeAttr (the TypeAttr counterpart of
// UnknownAttr), instead of being dropped, so it shows up in the declared type
// and round-trips under -ast-print.

// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -Wno-unknown-attributes -ast-print %s \
// RUN:   | FileCheck --check-prefix=PRINT %s

int *[[ns::transient(a, b)]] p;

// The retained attribute is part of the pointer's (sugared) type.
// CHECK: VarDecl {{.*}} p 'int * {{\[\[}}ns::transient(a, b){{\]\]}}':'int *'

// -ast-print reproduces the attribute and its arguments, but prints a trailing
// type attribute after the declarator, so the exact written position is not
// preserved (harmless for an ignored attribute; the content round-trips).
// PRINT: int *p {{\[\[}}ns::transient(a, b){{\]\]}};

// The wrapper survives template instantiation: TreeTransform rebuilds the
// AttributedType, so the specialization's member keeps the attribute with the
// substituted underlying type. This is the same path [[clang::annotate_type]]
// takes, so no special support is needed.
template <class T> struct S {
  T *[[ns::transient(a, b)]] q;
};
template struct S<int>;

// CHECK: ClassTemplateSpecializationDecl {{.*}} struct S definition
// CHECK: FieldDecl {{.*}} q 'int * {{\[\[}}ns::transient(a, b){{\]\]}}':'int *'
