// RUN: %clang_analyze_cc1 -analyzer-checker=unix.Malloc %s
// Do not crash due to division by zero

int f(unsigned int a) {
  if (a <= 0) return 1 / a;
  return a;
}
