
#define Outer(action) action

void completeParam(int param) {
    ;
    Outer(__extension__({ _Pragma("clang diagnostic push") }));
    param;
}

// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:7:1 %s | FileCheck %s
// CHECK: param : [#int#]param

void completeParamPragmaError(int param) {
    Outer(__extension__({ _Pragma(2) })); // expected-error {{_Pragma takes a parenthesized string literal}}
    param;
}

// RUN: %clang_cc1 -fsyntax-only -verify -code-completion-at=%s:16:1 %s | FileCheck %s
