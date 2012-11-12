namespace N {
  enum Color {
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Indigo,
    Violet
  };
}

void test(enum N::Color color) {
  switch (color) {
  case N::Red:
    break;
    
  case N::Yellow:
    break;
    
  case 
    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:21:8 %s -o - | FileCheck -check-prefix=CC1 %s
    // CHECK-CC1: Blue : [#N::Color#]N::Blue
    // CHECK-CC1-NEXT: Green : [#N::Color#]N::Green
    // CHECK-CC1-NEXT: Indigo : [#N::Color#]N::Indigo
    // CHECK-CC1-NEXT: Orange : [#N::Color#]N::Orange
    // CHECK-CC1-NEXT: Violet : [#N::Color#]N::Violet
    
