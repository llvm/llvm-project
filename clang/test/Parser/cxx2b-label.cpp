// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx23 -std=c++23 -Wpre-c++23-compat %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

void test_label_in_func() {
label1:
    int x;
label2:
    x = 1;
label3: label4: label5:
} // cxx20-warning {{label at end of compound statement is a C++23 extension}} \
     cxx23-warning {{label at end of compound statement is incompatible with C++ standards before C++23}}

int test_label_in_switch(int v) {
    switch (v) {
    case 1:
        return 1;
    case 2:
        return 2;
    case 3: case 4: case 5:
    } // cxx20-warning {{label at end of compound statement is a C++23 extension}} \
         cxx23-warning {{label at end of compound statement is incompatible with C++ standards before C++23}}

    switch (v) {
    case 6:
        return 6;
    default:
    } // cxx20-warning {{label at end of compound statement is a C++23 extension}} \
         cxx23-warning {{label at end of compound statement is incompatible with C++ standards before C++23}}

    return 0;
}
