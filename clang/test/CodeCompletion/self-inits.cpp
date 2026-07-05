int foo = 10;
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):11 %s -o - | FileCheck --check-prefix=CC1 %s
// CC1-NOT: foo
