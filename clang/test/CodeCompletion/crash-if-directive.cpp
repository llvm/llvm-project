#define FOO(X) X
#if FOO(
#elif FOO(
#endif
// RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-3):9 %s
// RUN: not %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-3):11 %s
