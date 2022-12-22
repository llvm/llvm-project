void test(bool x) {
  if (x) {}
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):7 %s | FileCheck %s
  // CHECK: PREFERRED-TYPE: _Bool

  while (x) {}
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):10 %s | FileCheck %s

  for (; x;) {}
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-1):10 %s | FileCheck %s

  // FIXME(ibiryukov): the condition in do-while is parsed as expression, so we
  // fail to detect it should be converted to bool.
  // do {} while (x);
}
