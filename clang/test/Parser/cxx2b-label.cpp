// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx2b -std=c++2b -Wpre-c++2b-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

void foo() {
label1:
    int x;
label2:
    x = 1;
label3: label4: label5:
} // cxx20-warning {{label at end of compound statement is a C++2b extension}} \
     cxx2b-warning {{label at end of compound statement is incompatible with C++ standards before C++2b}}
