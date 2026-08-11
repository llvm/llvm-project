// RUN: %clang_cc1 -verify -fopenmp -std=c++17 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++17 %s
// expected-no-diagnostics

// Regression test for https://github.com/llvm/llvm-project/issues/191549
// Inscan reductions with non-dependent types in function templates were
// incorrectly rejected because the DSA stack was not populated in
// dependent contexts.

template <typename T>
void exclusive_in_template() {
    int a[100], b[100];
    int sum = 0;
    #pragma omp parallel for reduction(inscan, +: sum)
    for (int i = 0; i < 100; i++) {
        a[i] = sum;
        #pragma omp scan exclusive(sum)
        sum += b[i];
    }
}

template <typename T>
void inclusive_in_template() {
    int a[100], b[100];
    int sum = 0;
    #pragma omp parallel for reduction(inscan, +: sum)
    for (int i = 0; i < 100; i++) {
        sum += b[i];
        #pragma omp scan inclusive(sum)
        a[i] = sum;
    }
}

template <typename T>
void for_simd_inscan_in_template() {
    int a[100], b[100];
    int sum = 0;
    #pragma omp parallel for simd reduction(inscan, +: sum)
    for (int i = 0; i < 100; i++) {
        a[i] = sum;
        #pragma omp scan exclusive(sum)
        sum += b[i];
    }
}

template <typename T>
void simd_inscan_in_template() {
    int a[100], b[100];
    int sum = 0;
    #pragma omp simd reduction(inscan, +: sum)
    for (int i = 0; i < 100; i++) {
        a[i] = sum;
        #pragma omp scan exclusive(sum)
        sum += b[i];
    }
}

void instantiate() {
    exclusive_in_template<int>();
    inclusive_in_template<int>();
    for_simd_inscan_in_template<int>();
    simd_inscan_in_template<int>();
}
