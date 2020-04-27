// RUN: rm -rf %t
// RUN: mkdir -p %t/ctudir
// RUN: cp %s %t/ctu-on-demand-parsing.cpp
// RUN: cp %S/ctu-hdr.h %t/ctu-hdr.h
// RUN: cp %S/Inputs/ctu-chain.cpp %t/ctudir/ctu-chain.cpp
// RUN: cp %S/Inputs/ctu-other.cpp %t/ctudir/ctu-other.cpp
// Path substitutions on Windows platform could contain backslashes. These are escaped in the json file.
// RUN: echo '[{"directory":"%t/ctudir","command":"clang++ -c ctu-chain.cpp","file":"ctu-chain.cpp"},{"directory":"%t/ctudir","command":"clang++ -c ctu-other.cpp","file":"ctu-other.cpp"}]' | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
// RUN: cd "%t/ctudir" && %clang_extdef_map ctu-chain.cpp ctu-other.cpp > externalDefMap.txt
// RUN: cd "%t" && %clang_analyze_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=ctudir \
// RUN:   -analyzer-config ctu-on-demand-parsing=true \
// RUN:   -verify ctu-on-demand-parsing.cpp
// RUN: cd "%t" && %clang_analyze_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir="%t/ctudir" \
// RUN:   -analyzer-config ctu-on-demand-parsing=true \
// RUN:   -analyzer-config display-ctu-progress=true 2>&1 ctu-on-demand-parsing.cpp | FileCheck %t/ctu-on-demand-parsing.cpp

// CHECK: CTU loaded AST file: {{.*}}ctu-other.cpp
// CHECK: CTU loaded AST file: {{.*}}ctu-chain.cpp

#include "ctu-hdr.h"

void clang_analyzer_eval(int);

int f(int);
int g(int);
int h(int);

int callback_to_main(int x) { return x + 1; }

namespace myns {
int fns(int x);

namespace embed_ns {
int fens(int x);
}

class embed_cls {
public:
  int fecl(int x);
};
} // namespace myns

class mycls {
public:
  int fcl(int x);
  virtual int fvcl(int x);
  static int fscl(int x);

  class embed_cls2 {
  public:
    int fecl2(int x);
  };
};

class derived : public mycls {
public:
  virtual int fvcl(int x) override;
};

namespace chns {
int chf1(int x);
}

int fun_using_anon_struct(int);
int other_macro_diag(int);

void test_virtual_functions(mycls *obj) {
  // The dynamic type is known.
  clang_analyzer_eval(mycls().fvcl(1) == 8);   // expected-warning{{TRUE}}
  clang_analyzer_eval(derived().fvcl(1) == 9); // expected-warning{{TRUE}}
  // We cannot decide about the dynamic type.
  clang_analyzer_eval(obj->fvcl(1) == 8); // expected-warning{{FALSE}} expected-warning{{TRUE}}
  clang_analyzer_eval(obj->fvcl(1) == 9); // expected-warning{{FALSE}} expected-warning{{TRUE}}
}

int main() {
  clang_analyzer_eval(f(3) == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(f(4) == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(f(5) == 3); // expected-warning{{FALSE}}
  clang_analyzer_eval(g(4) == 6); // expected-warning{{TRUE}}
  clang_analyzer_eval(h(2) == 8); // expected-warning{{TRUE}}

  clang_analyzer_eval(myns::fns(2) == 9);                   // expected-warning{{TRUE}}
  clang_analyzer_eval(myns::embed_ns::fens(2) == -1);       // expected-warning{{TRUE}}
  clang_analyzer_eval(mycls().fcl(1) == 6);                 // expected-warning{{TRUE}}
  clang_analyzer_eval(mycls::fscl(1) == 7);                 // expected-warning{{TRUE}}
  clang_analyzer_eval(myns::embed_cls().fecl(1) == -6);     // expected-warning{{TRUE}}
  clang_analyzer_eval(mycls::embed_cls2().fecl2(0) == -11); // expected-warning{{TRUE}}

  clang_analyzer_eval(chns::chf1(4) == 12);           // expected-warning{{TRUE}}
  clang_analyzer_eval(fun_using_anon_struct(8) == 8); // expected-warning{{TRUE}}

  clang_analyzer_eval(other_macro_diag(1) == 1); // expected-warning{{TRUE}}
  // expected-warning@ctudir/ctu-other.cpp:93{{REACHABLE}}
  MACRODIAG(); // expected-warning{{REACHABLE}}
}
