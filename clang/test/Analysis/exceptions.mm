// RUN: %clang_analyze_cc1 -fexceptions -fobjc-exceptions -fcxx-exceptions -analyzer-checker=core,unix.Malloc,debug.ExprInspection -verify %s

void clang_analyzer_checkInlined(bool);

typedef typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);


id getException();
void inlinedObjC() {
  clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
  @throw getException();
}

int testObjC() {
  int a; // uninitialized
  void *mem = malloc(4); // no-warning (ObjC exceptions are usually fatal)
  inlinedObjC();
  free(mem);
  return a; // no-warning
}


void inlinedCXX() {
  clang_analyzer_checkInlined(true); // expected-warning{{TRUE}}
  throw -1;
}

int testCXX() {
  int a; // uninitialized
  // FIXME: this should be reported as a leak, because C++ exceptions are
  // often not fatal.
  void *mem = malloc(4);
  inlinedCXX();
  free(mem);
  return a; // no-warning
}
