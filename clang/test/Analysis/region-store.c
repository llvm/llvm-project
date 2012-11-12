// RUN: %clang_cc1 -analyze -analyzer-checker=core,unix -verify %s
// expected-no-diagnostics

int printf(const char *restrict,...);

// Testing core functionality of the region store.
// radar://10127782
int compoundLiteralTest() {
    int index = 0;
    for (index = 0; index < 2; index++) {
        int thing = (int []){0, 1}[index];
        printf("thing: %i\n", thing);
    }
    return 0;
}

int compoundLiteralTest2() {
    int index = 0;
    for (index = 0; index < 3; index++) {
        int thing = (int [][3]){{0,0,0}, {1,1,1}, {2,2,2}}[index][index];
        printf("thing: %i\n", thing);
    }
    return 0;
}
