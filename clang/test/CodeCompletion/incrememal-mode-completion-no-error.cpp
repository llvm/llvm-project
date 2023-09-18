void foo(
// RUN: %clang_cc1 -fincremental-extensions -fsyntax-only -code-completion-at=%s:%(line-1):9 %s | wc -c | FileCheck %s
// CHECK: 0
