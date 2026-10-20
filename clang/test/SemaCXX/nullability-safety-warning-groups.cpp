// Tests for warning group suppression and control.
//
// -Wno-nullability-safety-dereference suppresses the warning:
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Wno-nullability-safety-dereference -verify=suppressed %s
//
// Parent group -Wno-nullability-safety also suppresses:
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Wno-nullability-safety -verify=suppressed %s
//
// -Werror=nullability-safety-dereference promotes to error:
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nullable -Werror=nullability-safety-dereference -verify=werror %s
//
// cc1 rejects invalid -fnullability-default value:
// RUN: not %clang_cc1 -fnullability-default=invalid %s 2>&1 | FileCheck %s
// CHECK: error: invalid value 'invalid' in '-fnullability-default=invalid'

// suppressed-no-diagnostics

void test(int * _Nullable p) {
    *p = 42; // werror-error {{dereference of nullable pointer}} werror-note {{add a null check}}
}
