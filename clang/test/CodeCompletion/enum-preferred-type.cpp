namespace N {
  enum Color {
    Red,
    Blue,
    Orange,
  };
}

void test(N::Color color) {
  color = N::Color::Red;
  test(N::Color::Red);
  if (color == N::Color::Red) {}
  // FIXME: ideally, we should not show 'Red' on the next line.
  else if (color == N::Color::Blue) {}

  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-6):11 %s -o - | FileCheck %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-6):8 %s -o - | FileCheck %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-6):16 %s -o - | FileCheck %s
  // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-5):21 %s -o - | FileCheck %s
  // CHECK: Blue : [#N::Color#]N::Blue
  // CHECK: color : [#N::Color#]color
  // CHECK: Orange : [#N::Color#]N::Orange
  // CHECK: Red : [#N::Color#]N::Red
}
