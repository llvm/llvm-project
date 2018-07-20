struct IncompleteType;
int int_value;
typedef int int_typedef;

void f(in);
IncompleteType g(in);
// Completing should produce results even if types are incomplete.
// Note that clang is expected to return an error code since 'in' does not resolve.
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:5:9 %s -o - | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:6:19 %s -o - | FileCheck %s
// CHECK: COMPLETION: int{{$}}
// CHECK: COMPLETION: int_typedef
// CHECK: COMPLETION: int_value
