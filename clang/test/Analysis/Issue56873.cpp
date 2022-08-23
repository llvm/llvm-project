// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

struct S {
};

void Issue56873_1() {
    int n;

    // This line used to crash
    S *arr = new S[n];
    
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}

void Issue56873_2() {
    int n;

    // This line used to crash
    int *arr = new int[n];
    
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}
