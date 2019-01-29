// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-other.cpp.ast %S/Inputs/ctu-other.cpp
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-chain.cpp.ast %S/Inputs/ctu-chain.cpp
// RUN: cp %S/Inputs/ctu-other.cpp.externalDefMap.txt %t/ctudir/externalDefMap.txt
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config display-ctu-progress=true 2>&1 %s | FileCheck %s

// CHECK: CTU loaded AST file: {{.*}}ctu-other.cpp.ast
// CHECK: CTU loaded AST file: {{.*}}ctu-chain.cpp.ast

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
}

class mycls {
public:
  int fcl(int x);
  static int fscl(int x);

  class embed_cls2 {
  public:
    int fecl2(int x);
  };
};

namespace chns {
int chf1(int x);
}

int fun_using_anon_struct(int);
int other_macro_diag(int);

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

  clang_analyzer_eval(chns::chf1(4) == 12); // expected-warning{{TRUE}}
  clang_analyzer_eval(fun_using_anon_struct(8) == 8); // expected-warning{{TRUE}}

  clang_analyzer_eval(other_macro_diag(1) == 1); // expected-warning{{TRUE}}
  // expected-warning@Inputs/ctu-other.cpp:80{{REACHABLE}}
  MACRODIAG(); // expected-warning{{REACHABLE}}
}
