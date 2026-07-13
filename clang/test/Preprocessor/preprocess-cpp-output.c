// RUN: %clang_cc1 -E -x c %s | FileCheck %s --check-prefixes=EXPANDED
// RUN: %clang_cc1 -E -x cpp-output %s | FileCheck %s --check-prefixes=NOT-EXPANDED

// EXPANDED: void __attribute__((__attribute__((always_inline)))) foo()
// NOT-EXPANDED: void __attribute__((always_inline)) foo()

#define always_inline __attribute__((always_inline))
void __attribute__((always_inline)) foo() {
    return 4;
}
