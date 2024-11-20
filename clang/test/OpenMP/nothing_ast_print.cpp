// RUN: %clang_cc1 -fopenmp -ast-print %s | FileCheck %s --check-prefix=PRINT
// RUN: %clang_cc1 -ast-print %s | FileCheck %s --check-prefix=PRINT

// Checks whether the `if` body looks same with and without OpenMP enabled

void foo() {
    return;
}

int main() {
    int x = 3;
    if (x % 2 == 0)
        #pragma omp nothing
    foo();

    return 0;
// PRINT: if (x % 2 == 0)
// PRINT:    foo();
// PRINT: return 0;
}