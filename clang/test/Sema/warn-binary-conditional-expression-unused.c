// RUN: %clang_cc1 -fsyntax-only -Wunused-value -verify %s
int main(void) {
    int a;
    int b;
    a ? : b; //expected-warning{{expression result unused}}
    a ? a : b; //expected-warning{{expression result unused}}
    a ? : ++b;
    a ? a : ++b;
    ++a ? : b; //expected-warning{{expression result unused}}
    ++a ? a : b; //expected-warning{{expression result unused}}
    ++a ? : ++b;
    ++a ? a : ++b;
    return 0;
};

