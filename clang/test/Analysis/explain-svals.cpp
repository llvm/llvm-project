// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -analyzer-checker=core.builtin,debug.ExprInspection,unix.cstring -verify %s

typedef unsigned long size_t;

struct S {
  struct S3 {
    int y[10];
  };
  struct S2 : S3 {
    int *x;
  } s2[10];
  int z;
};


void clang_analyzer_explain(int);
void clang_analyzer_explain(void *);
void clang_analyzer_explain(S);

size_t clang_analyzer_getExtent(void *);

size_t strlen(const char *);

int conjure();
S conjure_S();

int glob;
static int stat_glob;
void *glob_ptr;

// Test strings are regex'ed because we need to match exact string
// rather than a substring.

void test_1(int param, void *ptr) {
  clang_analyzer_explain(&glob); // expected-warning-re{{{{^pointer to global variable 'glob'$}}}}
  clang_analyzer_explain(param); // expected-warning-re{{{{^argument 'param'$}}}}
  clang_analyzer_explain(ptr); // expected-warning-re{{{{^argument 'ptr'$}}}}
  if (param == 42)
    clang_analyzer_explain(param); // expected-warning-re{{{{^signed 32-bit integer '42'$}}}}
}

void test_2(char *ptr, int ext) {
  clang_analyzer_explain((void *) "asdf"); // expected-warning-re{{{{^pointer to element of type 'char' with index 0 of string literal "asdf"$}}}}
  clang_analyzer_explain(strlen(ptr)); // expected-warning-re{{{{^metadata of type 'unsigned long' tied to pointee of argument 'ptr'$}}}}
  clang_analyzer_explain(conjure()); // expected-warning-re{{{{^symbol of type 'int' conjured at statement 'conjure\(\)'$}}}}
  clang_analyzer_explain(glob); // expected-warning-re{{{{^value derived from \(symbol of type 'int' conjured at statement 'conjure\(\)'\) for global variable 'glob'$}}}}
  clang_analyzer_explain(glob_ptr); // expected-warning-re{{{{^value derived from \(symbol of type 'int' conjured at statement 'conjure\(\)'\) for global variable 'glob_ptr'$}}}}
  clang_analyzer_explain(clang_analyzer_getExtent(ptr)); // expected-warning-re{{{{^extent of pointee of argument 'ptr'$}}}}
  int *x = new int[ext];
  clang_analyzer_explain(x); // expected-warning-re{{{{^pointer to element of type 'int' with index 0 of heap segment that starts at symbol of type 'int \*' conjured at statement 'new int \[ext\]'$}}}}
  // Sic! What gets computed is the extent of the element-region.
  clang_analyzer_explain(clang_analyzer_getExtent(x)); // expected-warning-re{{{{^signed 32-bit integer '4'$}}}}
  delete[] x;
}

void test_3(S s) {
  clang_analyzer_explain(&s); // expected-warning-re{{{{^pointer to parameter 's'$}}}}
  clang_analyzer_explain(s.z); // expected-warning-re{{{{^initial value of field 'z' of parameter 's'$}}}}
  clang_analyzer_explain(&s.s2[5].y[3]); // expected-warning-re{{{{^pointer to element of type 'int' with index 3 of field 'y' of base object 'S::S3' inside element of type 'struct S::S2' with index 5 of field 's2' of parameter 's'$}}}}
  if (!s.s2[7].x) {
    clang_analyzer_explain(s.s2[7].x); // expected-warning-re{{{{^concrete memory address '0'$}}}}
    // FIXME: we need to be explaining '1' rather than '0' here; not explainer bug.
    clang_analyzer_explain(s.s2[7].x + 1); // expected-warning-re{{{{^concrete memory address '0'$}}}}
  }
}

void test_4(int x, int y) {
  int z;
  static int stat;
  clang_analyzer_explain(x + 1); // expected-warning-re{{{{^\(argument 'x'\) \+ 1$}}}}
  clang_analyzer_explain(1 + y); // expected-warning-re{{{{^\(argument 'y'\) \+ 1$}}}}
  clang_analyzer_explain(x + y); // expected-warning-re{{{{^unknown value$}}}}
  clang_analyzer_explain(z); // expected-warning-re{{{{^undefined value$}}}}
  clang_analyzer_explain(&z); // expected-warning-re{{{{^pointer to local variable 'z'$}}}}
  clang_analyzer_explain(stat); // expected-warning-re{{{{^signed 32-bit integer '0'$}}}}
  clang_analyzer_explain(&stat); // expected-warning-re{{{{^pointer to static local variable 'stat'$}}}}
  clang_analyzer_explain(stat_glob); // expected-warning-re{{{{^initial value of global variable 'stat_glob'$}}}}
  clang_analyzer_explain(&stat_glob); // expected-warning-re{{{{^pointer to global variable 'stat_glob'$}}}}
  clang_analyzer_explain((int[]){1, 2, 3}); // expected-warning-re{{{{^pointer to element of type 'int' with index 0 of temporary object constructed at statement '\(int \[3\]\)\{1, 2, 3\}'$}}}}
}

namespace {
class C {
  int x[10];

public:
  void test_5(int i) {
    clang_analyzer_explain(this); // expected-warning-re{{{{^pointer to 'this' object$}}}}
    clang_analyzer_explain(&x[i]); // expected-warning-re{{{{^pointer to element of type 'int' with index 'argument 'i'' of field 'x' of 'this' object$}}}}
    clang_analyzer_explain(__builtin_alloca(i)); // expected-warning-re{{{{^pointer to region allocated by '__builtin_alloca\(i\)'$}}}}
  }
};
} // end of anonymous namespace

void test_6() {
  clang_analyzer_explain(conjure_S()); // expected-warning-re{{{{^lazily frozen compound value of temporary object constructed at statement 'conjure_S\(\)'$}}}}
  clang_analyzer_explain(conjure_S().z); // expected-warning-re{{{{^value derived from \(symbol of type 'struct S' conjured at statement 'conjure_S\(\)'\) for field 'z' of temporary object constructed at statement 'conjure_S\(\)'$}}}}
}
