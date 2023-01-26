enum Color {
  Red,
  Orange,
  Yellow,
  Green,
  Blue,
  Indigo,
  Violet
};

void test(enum Color color) {
  switch (color) {
    case Red:
      break;
      
    case Yellow:
      break;

    case Green:
      break;
  }

  unsigned c2;
  switch (c2) {
    case 
  }

    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-9):10 %s -o - | FileCheck -check-prefix=CHECK-CC1 %s
    // CHECK-CC1: Blue
    // CHECK-CC1-NEXT: Green
    // CHECK-CC1-NEXT: Indigo
    // CHECK-CC1-NEXT: Orange
    // CHECK-CC1-NEXT: Violet

    // RUN: %clang_cc1 -fsyntax-only -code-completion-at=%s:%(line-10):10 %s -o - | FileCheck -check-prefix=CHECK-CC2 %s      
  // CHECK-CC2: COMPLETION: Blue : [#enum Color#]Blue
  // CHECK-CC2-NEXT: COMPLETION: c2 : [#unsigned int#]c2
  // CHECK-CC2-NEXT: COMPLETION: color : [#enum Color#]color
  // CHECK-CC2-NEXT: COMPLETION: Green : [#enum Color#]Green
  // CHECK-CC2-NEXT: COMPLETION: Indigo : [#enum Color#]Indigo
  // CHECK-CC2-NEXT: COMPLETION: Orange : [#enum Color#]Orange
  // CHECK-CC2-NEXT: COMPLETION: Red : [#enum Color#]Red
  // CHECK-CC2-NEXT: COMPLETION: Pattern : [#size_t#]sizeof(<#expression-or-type#>)
  // CHECK-CC2-NEXT: COMPLETION: Violet : [#enum Color#]Violet
  // CHECK-CC2-NEXT: COMPLETION: Yellow : [#enum Color#]Yellow
