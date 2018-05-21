// RUN: %clang_cc1 -triple thumbv7-unknown-windows-msvc -fobjc-exceptions -fobjc-runtime=ios -verify %s

extern void g(void);
void f() {
  @try {
    g();
  } @catch (Class c) { // expected-error{{@catch parameter is not a pointer to an interface type}}
  }
}
