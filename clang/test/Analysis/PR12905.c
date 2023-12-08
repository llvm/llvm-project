// RUN: %clang_analyze_cc1 -analyzer-checker=core %s
// PR12905

void C(void);

void t(void) {
  C();
}
