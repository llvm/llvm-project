// Tests for warning group suppression and control.
//
// -Wno-flow-nullable-dereference suppresses the warning:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-flow-nullable-dereference -verify=suppressed %s
//
// Parent group -Wno-flow-nullability also suppresses:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-flow-nullability -verify=suppressed %s
//
// -Werror=flow-nullable-dereference promotes to error:
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Werror=flow-nullable-dereference -verify=werror %s
//
// cc1 rejects invalid -fnullability-default value:
// RUN: not %clang_cc1 -fnullability-default=invalid %s 2>&1 | FileCheck %s
// CHECK: error: invalid value 'invalid' in '-fnullability-default=invalid'

// suppressed-no-diagnostics

void test(int * _Nullable p) {
    *p = 42; // werror-error {{dereference of nullable pointer}} werror-note {{add a null check}}
}
