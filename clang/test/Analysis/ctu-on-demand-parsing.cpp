// RUN: rm -rf %t
// RUN: mkdir -p %t/Inputs
// RUN: cp %s %t/ctu-on-demand-parsing.cpp
// RUN: cp %S/ctu-hdr.h %t/ctu-hdr.h
// RUN: cp %S/Inputs/ctu-chain.cpp %t/Inputs/ctu-chain.cpp
// RUN: cp %S/Inputs/ctu-other.cpp %t/Inputs/ctu-other.cpp
//
// Path substitutions on Windows platform could contain backslashes. These are escaped in the json file.
// compile_commands.json is only needed for the extdef_mapping, not for the analysis itself.
// RUN: echo '[{"directory":"%t/Inputs","command":"clang++ ctu-chain.cpp","file":"ctu-chain.cpp"},{"directory":"%t/Inputs","command":"clang++ ctu-other.cpp","file":"ctu-other.cpp"}]' | sed -e 's/\\/\\\\/g' > %t/compile_commands.json
//
// RUN: echo '{"%t/Inputs/ctu-chain.cpp": ["g++", "%t/Inputs/ctu-chain.cpp"], "%t/Inputs/ctu-other.cpp": ["g++", "%t/Inputs/ctu-other.cpp"]}' | sed -e 's/\\/\\\\/g' > %t/invocations.yaml
//
// RUN: cd "%t" && %clang_extdef_map Inputs/ctu-chain.cpp Inputs/ctu-other.cpp > externalDefMap.txt
//
// RUN: cd "%t" && %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=. \
// RUN:   -analyzer-config ctu-invocation-list=invocations.yaml \
// RUN:   -verify ctu-on-demand-parsing.cpp
// RUN: cd "%t" && %clang_analyze_cc1 \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=. \
// RUN:   -analyzer-config ctu-invocation-list=invocations.yaml \
// RUN:   -analyzer-config display-ctu-progress=true ctu-on-demand-parsing.cpp 2>&1 | FileCheck %t/ctu-on-demand-parsing.cpp
//
// CHECK: CTU loaded AST file: {{.*}}ctu-other.cpp
// CHECK: CTU loaded AST file: {{.*}}ctu-chain.cpp
//
// FIXME: Path handling should work on all platforms.
// REQUIRES: system-linux

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
  // expected-warning@Inputs/ctu-other.cpp:93{{REACHABLE}}
  MACRODIAG(); // expected-warning{{REACHABLE}}
}
