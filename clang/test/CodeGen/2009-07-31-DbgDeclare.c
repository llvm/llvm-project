// RUN: %clang_cc1 -S -g -o %t.s %s
void foo() {
     int i = 0;
     i = 42;
}
