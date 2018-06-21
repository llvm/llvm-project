// RUN: echo "static void staticFunctionHeader(int i) {;}" > %T/header.h
// RUN: echo "static void staticFunctionHeader(int  /*i*/) {;}" > %T/header-fixed.h
// RUN: %check_clang_tidy %s misc-unused-parameters %t -- -header-filter='.*' -- -std=c++11 -fno-delayed-template-parsing
// RUN: diff %T/header.h %T/header-fixed.h

#include "header.h"
// CHECK-MESSAGES: header.h:1:38: warning

// Basic removal
// =============
void a(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void a(int  /*i*/) {;}{{$}}

void b(int i = 1) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void b(int  /*i*/ = 1) {;}{{$}}

void c(int *i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:13: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void c(int * /*i*/) {;}{{$}}

void d(int i[]) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void d(int  /*i*/[]) {;}{{$}}

void e(int i[1]) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: parameter 'i' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void e(int  /*i*/[1]) {;}{{$}}

void f(void (*fn)()) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: parameter 'fn' is unused [misc-unused-parameters]
// CHECK-FIXES: {{^}}void f(void (* /*fn*/)()) {;}{{$}}

// Unchanged cases
// ===============
void f(int i); // Don't remove stuff in declarations
void g(int i = 1);
void h(int i[]);
void s(int i[1]);
void u(void (*fn)());
void w(int i) { (void)i; } // Don't remove used parameters

bool useLambda(int (*fn)(int));
static bool static_var = useLambda([] (int a) { return a; });

// Remove parameters of local functions
// ====================================
static void staticFunctionA(int i);
// CHECK-FIXES: {{^}}static void staticFunctionA();
static void staticFunctionA(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning
// CHECK-FIXES: {{^}}static void staticFunctionA()

static void staticFunctionB(int i, int j) { (void)i; }
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning
// CHECK-FIXES: {{^}}static void staticFunctionB(int i)

static void staticFunctionC(int i, int j) { (void)j; }
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning
// CHECK-FIXES: {{^}}static void staticFunctionC(int j)

static void staticFunctionD(int i, int j, int k) { (void)i; (void)k; }
// CHECK-MESSAGES: :[[@LINE-1]]:40: warning
// CHECK-FIXES: {{^}}static void staticFunctionD(int i, int k)

static void staticFunctionE(int i = 4) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning
// CHECK-FIXES: {{^}}static void staticFunctionE()

static void staticFunctionF(int i = 4);
// CHECK-FIXES: {{^}}static void staticFunctionF();
static void staticFunctionF(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning
// CHECK-FIXES: {{^}}static void staticFunctionF()

static void staticFunctionG(int i[]);
// CHECK-FIXES: {{^}}static void staticFunctionG();
static void staticFunctionG(int i[]) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:33: warning
// CHECK-FIXES: {{^}}static void staticFunctionG()

static void staticFunctionH(void (*fn)());
// CHECK-FIXES: {{^}}static void staticFunctionH();
static void staticFunctionH(void (*fn)()) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:36: warning
// CHECK-FIXES: {{^}}static void staticFunctionH()

static void someCallSites() {
  staticFunctionA(1);
// CHECK-FIXES: staticFunctionA();
  staticFunctionB(1, 2);
// CHECK-FIXES: staticFunctionB(1);
  staticFunctionC(1, 2);
// CHECK-FIXES: staticFunctionC(2);
  staticFunctionD(1, 2, 3);
// CHECK-FIXES: staticFunctionD(1, 3);
  staticFunctionE(1);
// CHECK-FIXES: staticFunctionE();
  staticFunctionF(1);
// CHECK-FIXES: staticFunctionF();
  staticFunctionF();
// CHECK-FIXES: staticFunctionF();
  int t[] = {1};
  staticFunctionG(t);
// CHECK-FIXES: staticFunctionG();
  void func();
  staticFunctionH(&func);
// CHECK-FIXES: staticFunctionH();
}

/*
 * FIXME: This fails because the removals overlap and ClangTidy doesn't apply
 *        them.
 * static void bothVarsUnused(int a, int b) {;}
 */

// Regression test for long variable names and expressions
// =======================================================
static int variableWithLongName1(int LongName1, int LongName2) {
// CHECK-MESSAGES: :[[@LINE-1]]:53: warning: parameter 'LongName2' is unused
// CHECK-FIXES: {{^}}static int variableWithLongName1(int LongName1) {
  return LongName1;
}
static int variableWithLongName2(int LongName1, int LongName2) {
// CHECK-MESSAGES: :[[@LINE-1]]:38: warning: parameter 'LongName1' is unused
// CHECK-FIXES: {{^}}static int variableWithLongName2(int LongName2) {
  return LongName2;
}
static void someLongNameCallSites() {
  int LongName1 = 7, LongName2 = 17;
  variableWithLongName1(LongName1, LongName2);
// CHECK-FIXES: variableWithLongName1(LongName1);
  variableWithLongName2(LongName1, LongName2);
// CHECK-FIXES: variableWithLongName2(LongName2);
}

class SomeClass {
  static void f(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning
// CHECK-FIXES: static void f(int  /*i*/) {;}
  static void g(int i = 1) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning
// CHECK-FIXES: static void g(int  /*i*/ = 1) {;}
  static void h(int i[]) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning
// CHECK-FIXES: static void h(int  /*i*/[]) {;}
  static void s(void (*fn)()) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning
// CHECK-FIXES: static void s(void (* /*fn*/)()) {;}
};

namespace {
class C {
public:
  void f(int i);
// CHECK-FIXES: void f();
  void g(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning
// CHECK-FIXES: void g() {;}
  void h(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning
// CHECK-FIXES: void h(int  /*i*/) {;}
  void s(int i = 1) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning
// CHECK-FIXES: void s(int  /*i*/ = 1) {;}
  void u(int i[]) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning
// CHECK-FIXES: void u(int  /*i*/[]) {;}
  void w(void (*fn)()) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:17: warning
// CHECK-FIXES: void w(void (* /*fn*/)()) {;}
};

void C::f(int i) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning
// CHECK-FIXES: void C::f() {;}

template <typename T>
void useFunction(T t);

void someMoreCallSites() {
  C c;
  c.f(1);
// CHECK-FIXES: c.f();
  c.g(1);
// CHECK-FIXES: c.g();

  useFunction(&C::h);
  useFunction(&C::s);
  useFunction(&C::u);
  useFunction(&C::w);
}

class Base {
  virtual void f(int i);
};

class Derived : public Base {
  void f(int i) override {;}
// CHECK-MESSAGES: :[[@LINE-1]]:14: warning
// CHECK-FIXES: void f(int  /*i*/) override {;}
};

} // end namespace

template <typename T> void someFunctionTemplate(T b, T e) { (void)b; (void)e; }

template <typename T> void someFunctionTemplateOneUnusedParam(T b, T e) { (void)e; }
// CHECK-MESSAGES: :[[@LINE-1]]:65: warning
// CHECK-FIXES: {{^}}template <typename T> void someFunctionTemplateOneUnusedParam(T  /*b*/, T e) { (void)e; }

template <typename T> void someFunctionTemplateAllUnusedParams(T b, T e) {;}
// CHECK-MESSAGES: :[[@LINE-1]]:66: warning
// CHECK-MESSAGES: :[[@LINE-2]]:71: warning
// CHECK-FIXES: {{^}}template <typename T> void someFunctionTemplateAllUnusedParams(T  /*b*/, T  /*e*/) {;}

static void dontGetConfusedByParametersInFunctionTypes() { void (*F)(int i); }

template <typename T> class Function {};
static Function<void(int, int i)> dontGetConfusedByFunctionReturnTypes() {
  return Function<void(int, int)>();
}

// Do not warn on empty function bodies.
void f(int foo) {}
