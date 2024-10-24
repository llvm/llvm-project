// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir2
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir2/ctu-other.c.ast %S/Inputs/ctu-other.c
// RUN: cp %S/Inputs/ctu-other.c.externalDefMap.ast-dump.txt %t/ctudir2/externalDefMap.txt

// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c89 -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir2 \
// RUN:   -analyzer-config ctu-phase1-inlining=none \
// RUN:   -verify=newctu %s

// Simulate the behavior of the previous CTU implementation by inlining all
// functions during the first phase. This way, the second phase is a noop.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -std=c89 -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir2 \
// RUN:   -analyzer-config ctu-phase1-inlining=all \
// RUN:   -verify=oldctu %s

void clang_analyzer_eval(int);

// A function that's definition is unknown both for single-tu (stu) and ctu
// mode.
int unknown(int);
void test_unknown() {
  int res = unknown(6);
  clang_analyzer_eval(res == 6); // newctu-warning{{UNKNOWN}}
                                 // oldctu-warning@-1{{UNKNOWN}}
}

// Test typedef and global variable in function.
typedef struct {
  int a;
  int b;
} FooBar;
extern FooBar fb;
int f(int);
void testGlobalVariable() {
  clang_analyzer_eval(f(5) == 1);         // newctu-warning{{TRUE}} ctu
                                          // newctu-warning@-1{{UNKNOWN}} stu
                                          // oldctu-warning@-2{{TRUE}}
}

// Test enums.
int enumCheck(void);
enum A { x,
         y,
         z };
void testEnum(void) {
  clang_analyzer_eval(x == 0);            // newctu-warning{{TRUE}}
                                          // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(enumCheck() == 42); // newctu-warning{{TRUE}} ctu
                                          // newctu-warning@-1{{UNKNOWN}} stu
                                          // oldctu-warning@-2{{TRUE}}
}

// Test that asm import does not fail.
int inlineAsm(void);
int testInlineAsm(void) {
  return inlineAsm();
}

// Test reporting error in a macro.
struct S;
int g(struct S *);
void testMacro(void) {
  g(0); // newctu-warning@Inputs/ctu-other.c:29 {{Access to field 'a' results in a dereference of a null pointer (loaded from variable 'ctx')}}
        // oldctu-warning@Inputs/ctu-other.c:29 {{Access to field 'a' results in a dereference of a null pointer (loaded from variable 'ctx')}}
}

// The external function prototype is incomplete.
// warning:implicit functions are prohibited by c99
void testImplicit(void) {
  int res = identImplicit(6);   // external implicit functions are not inlined
  clang_analyzer_eval(res == 6); // newctu-warning{{TRUE}} ctu
                                 // newctu-warning@-1{{UNKNOWN}} stu
                                 // oldctu-warning@-2{{TRUE}}
  // Call something with uninitialized from the same function in which the implicit was called.
  // This is necessary to reproduce a special bug in NoStoreFuncVisitor.
  int uninitialized;
  h(uninitialized); // newctu-warning{{1st function call argument is an uninitialized value}}
                    // oldctu-warning@-1{{1st function call argument is an uninitialized value}}
}

// Tests the import of functions that have a struct parameter
// defined in its prototype.
struct DataType {
  int a;
  int b;
};
int structInProto(struct DataType *d);
void testStructDefInArgument(void) {
  struct DataType d;
  d.a = 1;
  d.b = 0;
  // Not imported, thus remains unknown both in stu and ctu.
  clang_analyzer_eval(structInProto(&d) == 0); // newctu-warning{{UNKNOWN}}
                                               // oldctu-warning@-1{{UNKNOWN}}
}

int switchWithoutCases(int);
void testSwitchStmtCrash(int x) {
  switchWithoutCases(x);
}
