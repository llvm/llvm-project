// RUN: %clang_cc1 %s -verify -fno-builtin

#define _diagnose_if(...) __attribute__((diagnose_if(__VA_ARGS__)))

void failure() _diagnose_if(); // expected-error{{exactly 3 arguments}}
void failure() _diagnose_if(0); // expected-error{{exactly 3 arguments}}
void failure() _diagnose_if(0, ""); // expected-error{{exactly 3 arguments}}
void failure() _diagnose_if(0, "", "error", 1); // expected-error{{exactly 3 arguments}}
void failure() _diagnose_if(0, 0, "error"); // expected-error{{requires a string}}
void failure() _diagnose_if(0, "", "invalid"); // expected-error{{invalid diagnostic type for 'diagnose_if'; use "error" or "warning" instead}}
void failure() _diagnose_if(0, "", "ERROR"); // expected-error{{invalid diagnostic type}}
void failure(int a) _diagnose_if(a, "", ""); // expected-error{{invalid diagnostic type}}
void failure() _diagnose_if(a, "", ""); // expected-error{{undeclared identifier 'a'}}

int globalVar;
void never_constant() _diagnose_if(globalVar, "", "error"); // expected-error{{'diagnose_if' attribute expression never produces a constant expression}} expected-note{{subexpression not valid}}
void never_constant() _diagnose_if(globalVar, "", "warning"); // expected-error{{'diagnose_if' attribute expression never produces a constant expression}} expected-note{{subexpression not valid}}

int alwaysok(int q) _diagnose_if(0, "", "error");
int neverok(int q) _diagnose_if(1, "oh no", "error"); // expected-note 5{{from 'diagnose_if' attribute on 'neverok'}}
int alwayswarn(int q) _diagnose_if(1, "oh no", "warning"); // expected-note 5{{from 'diagnose_if' attribute}}
int neverwarn(int q) _diagnose_if(0, "", "warning");

void runConstant() {
  int m;
  alwaysok(0);
  alwaysok(1);
  alwaysok(m);

  {
    int (*pok)(int) = alwaysok;
    pok = &alwaysok;
  }

  neverok(0); // expected-error{{oh no}}
  neverok(1); // expected-error{{oh no}}
  neverok(m); // expected-error{{oh no}}
  {
    int (*pok)(int) = neverok; // expected-error{{oh no}}
    pok = &neverok; // expected-error{{oh no}}
  }

  alwayswarn(0); // expected-warning{{oh no}}
  alwayswarn(1); // expected-warning{{oh no}}
  alwayswarn(m); // expected-warning{{oh no}}
  {
    int (*pok)(int) = alwayswarn; // expected-warning{{oh no}}
    pok = &alwayswarn; // expected-warning{{oh no}}
  }

  neverwarn(0);
  neverwarn(1);
  neverwarn(m);
  {
    int (*pok)(int) = neverwarn;
    pok = &neverwarn;
  }
}

int abs(int q) _diagnose_if(q >= 0, "redundant abs call", "error"); //expected-note{{from 'diagnose_if'}}
void runVariable() {
  int m;
  abs(-1);
  abs(1); // expected-error{{redundant abs call}}
  abs(m);

  int (*pabs)(int) = abs;
  pabs = &abs;
}

#define _overloadable __attribute__((overloadable))

int ovl1(const char *n) _overloadable _diagnose_if(n, "oh no", "error"); // expected-note{{from 'diagnose_if'}}
int ovl1(void *m) _overloadable;

int ovl2(const char *n) _overloadable _diagnose_if(n, "oh no", "error"); // expected-note{{candidate function}}
int ovl2(char *m) _overloadable; // expected-note{{candidate function}}
void overloadsYay() {
  ovl1((void *)0);
  ovl1(""); // expected-error{{oh no}}

  ovl2((void *)0); // expected-error{{ambiguous}}
}

void errorWarnDiagnose1() _diagnose_if(1, "oh no", "error") // expected-note{{from 'diagnose_if'}}
  _diagnose_if(1, "nop", "warning");
void errorWarnDiagnose2() _diagnose_if(1, "oh no", "error") // expected-note{{from 'diagnose_if'}}
  _diagnose_if(1, "nop", "error");
void errorWarnDiagnose3() _diagnose_if(1, "nop", "warning")
  _diagnose_if(1, "oh no", "error"); // expected-note{{from 'diagnose_if'}}

void errorWarnDiagnoseArg1(int a) _diagnose_if(a == 1, "oh no", "error") // expected-note{{from 'diagnose_if'}}
  _diagnose_if(a == 1, "nop", "warning");
void errorWarnDiagnoseArg2(int a) _diagnose_if(a == 1, "oh no", "error") // expected-note{{from 'diagnose_if'}}
  _diagnose_if(a == 1, "nop", "error");
void errorWarnDiagnoseArg3(int a) _diagnose_if(a == 1, "nop", "warning")
  _diagnose_if(a == 1, "oh no", "error"); // expected-note{{from 'diagnose_if'}}

void runErrorWarnDiagnose() {
  errorWarnDiagnose1(); // expected-error{{oh no}}
  errorWarnDiagnose2(); // expected-error{{oh no}}
  errorWarnDiagnose3(); // expected-error{{oh no}}

  errorWarnDiagnoseArg1(1); // expected-error{{oh no}}
  errorWarnDiagnoseArg2(1); // expected-error{{oh no}}
  errorWarnDiagnoseArg3(1); // expected-error{{oh no}}
}

void warnWarnDiagnose() _diagnose_if(1, "oh no!", "warning") _diagnose_if(1, "foo", "warning"); // expected-note 2{{from 'diagnose_if'}}
void runWarnWarnDiagnose() {
  warnWarnDiagnose(); // expected-warning{{oh no!}} expected-warning{{foo}}
}

void declsStackErr1(int a) _diagnose_if(a & 1, "decl1", "error"); // expected-note 2{{from 'diagnose_if'}}
void declsStackErr1(int a) _diagnose_if(a & 2, "decl2", "error"); // expected-note{{from 'diagnose_if'}}
void declsStackErr2();
void declsStackErr2() _diagnose_if(1, "complaint", "error"); // expected-note{{from 'diagnose_if'}}
void declsStackErr3() _diagnose_if(1, "complaint", "error"); // expected-note{{from 'diagnose_if'}}
void declsStackErr3();
void runDeclsStackErr() {
  declsStackErr1(0);
  declsStackErr1(1); // expected-error{{decl1}}
  declsStackErr1(2); // expected-error{{decl2}}
  declsStackErr1(3); // expected-error{{decl1}}
  declsStackErr2(); // expected-error{{complaint}}
  declsStackErr3(); // expected-error{{complaint}}
}

void declsStackWarn1(int a) _diagnose_if(a & 1, "decl1", "warning"); // expected-note 2{{from 'diagnose_if'}}
void declsStackWarn1(int a) _diagnose_if(a & 2, "decl2", "warning"); // expected-note 2{{from 'diagnose_if'}}
void declsStackWarn2();
void declsStackWarn2() _diagnose_if(1, "complaint", "warning"); // expected-note{{from 'diagnose_if'}}
void declsStackWarn3() _diagnose_if(1, "complaint", "warning"); // expected-note{{from 'diagnose_if'}}
void declsStackWarn3();
void runDeclsStackWarn() {
  declsStackWarn1(0);
  declsStackWarn1(1); // expected-warning{{decl1}}
  declsStackWarn1(2); // expected-warning{{decl2}}
  declsStackWarn1(3); // expected-warning{{decl1}} expected-warning{{decl2}}
  declsStackWarn2(); // expected-warning{{complaint}}
  declsStackWarn3(); // expected-warning{{complaint}}
}

void noMsg(int n) _diagnose_if(n, "", "warning"); // expected-note{{from 'diagnose_if'}}
void runNoMsg() {
  noMsg(1); // expected-warning{{<no message provided>}}
}

void alwaysWarnWithArg(int a) _diagnose_if(1 || a, "alwaysWarn", "warning"); // expected-note{{from 'diagnose_if'}}
void runAlwaysWarnWithArg(int a) {
  alwaysWarnWithArg(a); // expected-warning{{alwaysWarn}}
}

// Test that diagnose_if warnings generated in system headers are not ignored.
#include "Inputs/diagnose-if-warn-system-header.h"
