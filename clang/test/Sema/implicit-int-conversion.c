// RUN: %clang_cc1 %s -verify -Wconversion -Wno-implicit-int-conversion -DSMALL=char -DBIG=int -DNO_DIAG
// RUN: %clang_cc1 %s -verify -Wno-conversion -Wimplicit-int-conversion -DSMALL=char -DBIG=int
// RUN: %clang_cc1 %s -verify -Wconversion -Wno-implicit-float-conversion -DSMALL=float -DBIG=double -DNO_DIAG
// RUN: %clang_cc1 %s -verify -Wno-conversion -Wimplicit-float-conversion -DSMALL=float -DBIG=double

void f(void) {
  SMALL a;
  BIG b = 0;
  a = b;
#ifndef NO_DIAG
  // expected-warning@-2 {{implicit conversion}}
#else
  // expected-no-diagnostics
#endif
}
