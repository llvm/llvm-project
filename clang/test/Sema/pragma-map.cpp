// RUN: %clang_cc1 %s -triple s390x-ibm-zos -fsyntax-only -verify
// REQUIRES: systemz-registered-target

// Testing missing declarations.
#pragma map(d0, "D0") // expected-warning{{failed to resolve '#pragma map' to a declaration}}
#pragma map(f0, "F0I") // expected-warning{{failed to resolve '#pragma map' to a declaration}}
#pragma map(f3, "F3DD") // expected-warning{{failed to resolve '#pragma map' to a declaration}}

// Testing pragma map after decl.
static void sf0(void);
static int s0;
#pragma map(sf0, "SF0") // expected-warning{{failed to resolve '#pragma map' to a declaration}}
#pragma map(s0, "S0") // expected-warning{{#pragma map is applicable to symbols with external linkage only; not applied to 's0'}}

#pragma map(sf1, "SF1") // expected-warning{{failed to resolve '#pragma map' to a declaration}}
#pragma map(s1, "S1") // expected-warning{{#pragma map is applicable to symbols with external linkage only; not applied to 's1'}}
static void sf1(void);
static int s1;

// Testing pragma map after decl and usage.
extern "C" void f4(void);
extern "C" double cos(double);
double cos(double, double);

void t0() {
  f4();
  double d = cos(3.14, 1.414);
}
#pragma map(f4, "F4") // expected-warning{{#pragma map can only applied to unused symbols}}
#pragma map(cos, "COS")

namespace N0 {
extern "C" void f6(void) {}
#pragma map(f6, "N0F6") // expected-warning{{#pragma map can only applied to unused symbols}}
}


void func1(int);
#pragma map(func1, "FUNC1")
extern "C" void func1(double);

void call_func1() {
  func1(1);
}
void func1(int) { }
void func1(double) { }

// Pragma between  declarations

#pragma map(func2, "FUNC2")
void func2(int);
extern "C" void func2(double);

void call_func2() {
  func2(1);
}
void func2(int) { }
void func2(double) { }

// Pragma after  declarations

void func3(int);
extern "C" void func3(double);
#pragma map(func3, "FUNC3")

void call_func3() {
  func3(1);
}
void func3(int) { }
void func3(double) { }

static void func4(int);
#pragma map(func4, "FUNC4")
extern "C" void func4(double);
void func4(int) { }
void call_func4() {
  func4(1);
  func4(1.0);
}
