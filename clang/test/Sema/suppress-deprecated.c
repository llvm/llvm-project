// RUN: %clang_cc1 -fsyntax-only -Wno-deprecated-declarations -verify %s
// expected-no-diagnostics
extern void OldFunction(void) __attribute__((deprecated));

int main (int argc, const char * argv[]) {
  OldFunction();
}

