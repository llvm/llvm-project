// RUN: %clang_analyze_cc1 -Wno-incompatible-library-redeclaration -analyzer-checker=core,unix.cstring,alpha.unix.cstring -verify %s

// expected-no-diagnostics

void *strcpy(unsigned char *, unsigned char *);

unsigned char a, b;
void testUnsignedStrcpy(void) {
  strcpy(&a, &b);
}
