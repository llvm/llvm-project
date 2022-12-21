// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s
// expected-no-diagnostics

namespace bbi_77010 {
int crash_NE(int rhs, int lhs, int x) {
    int band = lhs & rhs;
    if (0 <= band) {}
    if (rhs > 0) {}
    return band != x; // no-crash D112621
}
} // namespace bbi_77010
