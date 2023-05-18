// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s
// XFAIL: *

void clang_analyzer_eval(int);

extern const int extInt;

int main()
{
    clang_analyzer_eval(extInt == 2); // expected-warning{{TRUE}}
}

extern const int extInt = 2;
