// RUN: %clang_cc1 -fsyntax-only -std=c17 -Wc2x-compat -verify=c17 %s
// RUN: %clang_cc1 -fsyntax-only -std=c2x -Wpre-c2x-compat -verify=c2x %s

void test_label_in_func() {
    int x;
label1:
    x = 1;
label2: label3: label4:
} // c17-warning {{label at end of compound statement is a C23 extension}} \
     c2x-warning {{label at end of compound statement is incompatible with C standards before C23}}

int test_label_in_switch(int v) {
    switch (v) {
    case 1:
        return 1;
    case 2:
        return 2;
    case 3: case 4: case 5:
    } // c17-warning {{label at end of compound statement is a C23 extension}} \
         c2x-warning {{label at end of compound statement is incompatible with C standards before C23}}

    switch (v) {
    case 6:
        return 6;
    default:
    } // c17-warning {{label at end of compound statement is a C23 extension}} \
         c2x-warning {{label at end of compound statement is incompatible with C standards before C23}}

    return 0;
}
