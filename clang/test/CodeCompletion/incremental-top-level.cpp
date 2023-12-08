int foo = 10;
f
// RUN: %clang_cc1 -fincremental-extensions -fsyntax-only -code-completion-at=%s:%(line-1):1 %s | FileCheck %s
// CHECK: COMPLETION: foo : [#int#]foo
