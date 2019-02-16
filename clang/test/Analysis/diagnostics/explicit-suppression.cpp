// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config suppress-c++-stdlib=false -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config suppress-c++-stdlib=true -DSUPPRESSED=1 -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -DSUPPRESSED=1 -verify %s

#ifdef SUPPRESSED
// expected-no-diagnostics
#endif

#include "../Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

class C {
  // The virtual function is to make C not trivially copy assignable so that we call the
  // variant of std::copy() that does not defer to memmove().
  virtual int f();
};

void testCopyNull(C *I, C *E) {
  std::copy(I, E, (C *)0);
#ifndef SUPPRESSED
  // expected-warning@../Inputs/system-header-simulator-cxx.h:677 {{Called C++ object pointer is null}}
#endif
}
