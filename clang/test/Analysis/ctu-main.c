// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir2
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir2/ctu-other.c.ast %S/Inputs/ctu-other.c
// RUN: cp %S/Inputs/ctu-other.c.externalDefMap.ast-dump.txt %t/ctudir2/externalDefMap.txt
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -std=c89 -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir2 \
// RUN:   -verify %s

void clang_analyzer_eval(int);

// Test typedef and global variable in function.
typedef struct {
  int a;
  int b;
} FooBar;
extern FooBar fb;
int f(int);
void testGlobalVariable() {
  clang_analyzer_eval(f(5) == 1);         // expected-warning{{TRUE}}
}

// Test enums.
int enumCheck(void);
enum A { x,
         y,
         z };
void testEnum() {
  clang_analyzer_eval(x == 0);            // expected-warning{{TRUE}}
  clang_analyzer_eval(enumCheck() == 42); // expected-warning{{TRUE}}
}

// Test that asm import does not fail.
int inlineAsm();
int testInlineAsm() {
  return inlineAsm();
}

// Test reporting error in a macro.
struct S;
int g(struct S *);
void testMacro(void) {
  g(0); // expected-warning@Inputs/ctu-other.c:29 {{Access to field 'a' results in a dereference of a null pointer (loaded from variable 'ctx')}}
}

// The external function prototype is incomplete.
// warning:implicit functions are prohibited by c99
void testImplicit() {
  int res = identImplicit(6);   // external implicit functions are not inlined
  clang_analyzer_eval(res == 6); // expected-warning{{TRUE}}
  // Call something with uninitialized from the same function in which the implicit was called.
  // This is necessary to reproduce a special bug in NoStoreFuncVisitor.
  int uninitialized;
  h(uninitialized); // expected-warning{{1st function call argument is an uninitialized value}}
}

// Tests the import of functions that have a struct parameter
// defined in its prototype.
struct DataType {
  int a;
  int b;
};
int structInProto(struct DataType *d);
void testStructDefInArgument() {
  struct DataType d;
  d.a = 1;
  d.b = 0;
  clang_analyzer_eval(structInProto(&d) == 0); // expected-warning{{TRUE}} expected-warning{{FALSE}}
}
