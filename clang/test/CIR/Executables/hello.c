// RUN: %clang -fclangir -fclangir-direct-lowering -o %t %s
// RUN: %t | FileCheck %s
// XFAIL: *

int printf(const char *format);

int main (void) {
    printf ("Hello, world!\n");
    // CHECK: Hello, world!
    return 0;
}
