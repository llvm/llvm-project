// RUN: %clang_cc1 -fsyntax-only -std=c17 -Wc2x-compat -verify=c17 %s
// RUN: %clang_cc1 -fsyntax-only -std=c2x -Wpre-c2x-compat -verify=c2x %s

void foo() {
    int x;
label1:
    x = 1;
label2: label3: label4:
} // c17-warning {{label at end of compound statement is a C2x extension}} \
     c2x-warning {{label at end of compound statement is incompatible with C standards before C2x}}
