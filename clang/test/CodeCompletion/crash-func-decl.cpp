// Important that BB is unknown.
// This triggers completion in PCC_RecoveryInFunction context, with no function.
int AA(BB cc);
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:3:12 %s | FileCheck %s
// CHECK: COMPLETION: char
