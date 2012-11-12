// RUN: %clang_cc1 -analyze -analyzer-checker=core %s
// PR12905

void C(void);

void t(void) {
  C();
}
