// RUN: %clang_cc1 -fsyntax-only -flifetime-safety-inference -fexperimental-lifetime-safety-tu-analysis -Wlifetime-safety -Wno-dangling -I%S -verify %s

#include "Inputs/lifetime-analysis.h"

struct View;
struct [[gsl::Owner]] MyObj {
  View getView() const [[clang::lifetimebound]];
};
struct [[gsl::Pointer()]] View {
  View(const MyObj&);
};

// This would normally trigger a suggestion warning if -Wlifetime-safety-suggestions was on.
// Since it is off, we expect NO warnings or notes here.
// expected-no-diagnostics
View return_view_directly(View a) {
  return a; 
}
