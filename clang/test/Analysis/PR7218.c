// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-store region -verify %s
char PR7218(char a) {
    char buf[2];
    buf[0] = a;
    return buf[1]; // expected-warning {{Undefined or garbage value returned to caller}}
}
