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
