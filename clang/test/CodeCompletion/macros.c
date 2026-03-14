enum Color {
  Red, Green, Blue
};

struct Point {
  float x, y, z;
  enum Color color;
};

void test(struct Point *p) {
  // RUN: %clang_cc1 -include %S/Inputs/macros.h -fsyntax-only -code-completion-macros -code-completion-at=%s:%(line+1):14 %s -o - | FileCheck -check-prefix=CC1 %s
  switch (p->IDENTITY(color)) {
  // RUN: %clang_cc1 -include %S/Inputs/macros.h -fsyntax-only -code-completion-macros -code-completion-at=%s:%(line+1):10 %s -o - | FileCheck -check-prefix=CC2 %s
    case 
  }
  // RUN: %clang_cc1 -include %S/Inputs/macros.h -fsyntax-only -code-completion-macros -code-completion-at=%s:%(line+1):8 %s -o - | FileCheck -check-prefix=CC3 %s
#ifdef Q
#endif

  // Run the same tests, this time with macros loaded from the PCH file.
  // RUN: %clang_cc1 -emit-pch -o %t %S/Inputs/macros.h
  // RUN: %clang_cc1 -include-pch %t -fsyntax-only -code-completion-macros -code-completion-at=%s:%(line-10):14 %s -o - | FileCheck -check-prefix=CC1 %s
  // RUN: %clang_cc1 -include-pch %t -fsyntax-only -code-completion-macros -code-completion-at=%s:%(line-9):10 %s -o - | FileCheck -check-prefix=CC2 %s
  // RUN: %clang_cc1 -include-pch %t -fsyntax-only -code-completion-macros -code-completion-at=%s:%(line-7):8 %s -o - | FileCheck -check-prefix=CC3 %s

  // CC1: color
  // CC1: x
  // CC1: y
  // CC1: z

  // CC2: BAR(<#X#>, <#Y#>)
  // CC2: Blue
  // CC2: FOO
  // CC2: Green
  // CC2: IDENTITY(<#X#>)
  // CC2: MACRO_WITH_HISTORY(<#X#>, <#Y#>)
  // CC2: Red
  // CC2: WIBBLE

  // CC3: BAR
  // CC3: DEAD_MACRO
  // CC3: FOO
  // CC3: IDENTITY
  // CC3: MACRO_WITH_HISTORY
  // CC3: WIBBLE
}
