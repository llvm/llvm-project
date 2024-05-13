// Note: the run lines follow their respective tests, since line/column
// matter in this test.

#include "comments.h"

struct A {
  // <- code completion
  /* <- code completion */
};

// RUN: %clang_cc1 -I %S/Inputs -fsyntax-only -code-completion-at=%s:%(line-4):6 %s
// RUN: %clang_cc1 -I %S/Inputs -fsyntax-only -code-completion-at=%s:%(line-4):6 %s
// RUN: %clang_cc1 -I %S/Inputs -fsyntax-only -code-completion-at=%S/Inputs/comments.h:3:6 %s
