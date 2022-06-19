// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-other.cpp.ast %S/Inputs/ctu-other.cpp
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-chain.cpp.ast %S/Inputs/ctu-chain.cpp
// RUN: cp %S/Inputs/ctu-other.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config ctu-phase1-inlining=none \
// RUN:   -verify=newctu %s

// Simulate the behavior of the previous CTU implementation by inlining all
// functions during the first phase. This way, the second phase is a noop.
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config ctu-phase1-inlining=all \
// RUN:   -verify=oldctu %s

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
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

extern const int extInt;
namespace intns {
extern const int extInt;
}
struct S {
  int a;
};
extern const S extS;
extern S extNonConstS;
struct NonTrivialS {
  int a;
  // User declaring a dtor makes it non-trivial.
  ~NonTrivialS();
};
extern const NonTrivialS extNTS;
extern const int extHere;
const int extHere = 6;
struct A {
  static const int a;
};
struct SC {
  const int a;
};
extern const SC extSC;
struct ST {
  static const struct SC sc;
};
struct SCNest {
  struct SCN {
    const int a;
  } scn;
};
extern SCNest extSCN;
extern const SCNest::SCN extSubSCN;
struct SCC {
  SCC(int c);
  const int a;
};
extern SCC extSCC;
union U {
  const int a;
  const unsigned int b;
};
extern const U extU;

void test_virtual_functions(mycls* obj) {
  // The dynamic type is known.
  clang_analyzer_eval(mycls().fvcl(1) == 8);   // newctu-warning{{TRUE}} ctu
                                               // newctu-warning@-1{{UNKNOWN}} stu
                                               // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(derived().fvcl(1) == 9); // newctu-warning{{TRUE}} ctu
                                               // newctu-warning@-1{{UNKNOWN}} stu
                                               // oldctu-warning@-2{{TRUE}}
  // We cannot decide about the dynamic type.
  clang_analyzer_eval(obj->fvcl(1) == 8);      // newctu-warning{{TRUE}} ctu
                                               // newctu-warning@-1{{UNKNOWN}} ctu, stu
                                               // oldctu-warning@-2{{TRUE}}
                                               // oldctu-warning@-3{{UNKNOWN}}
}

class TestAnonUnionUSR {
public:
  inline float f(int value) {
    union {
      float f;
      int i;
    };
    i = value;
    return f;
  }
  static const int Test;
};

extern int testImportOfIncompleteDefaultParmDuringImport(int);

extern int testImportOfDelegateConstructor(int);

int main() {
  clang_analyzer_eval(f(3) == 2); // newctu-warning{{TRUE}} ctu
                                  // newctu-warning@-1{{UNKNOWN}} stu
                                  // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(f(4) == 3); // newctu-warning{{TRUE}} ctu
                                  // newctu-warning@-1{{UNKNOWN}} stu
                                  // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(f(5) == 3); // newctu-warning{{FALSE}} ctu
                                  // newctu-warning@-1{{UNKNOWN}} stu
                                  // oldctu-warning@-2{{FALSE}}
  clang_analyzer_eval(g(4) == 6); // newctu-warning{{TRUE}} ctu
                                  // newctu-warning@-1{{UNKNOWN}} stu
                                  // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(h(2) == 8); // newctu-warning{{TRUE}} ctu
                                  // newctu-warning@-1{{UNKNOWN}} stu
                                  // oldctu-warning@-2{{TRUE}}

  clang_analyzer_eval(myns::fns(2) == 9);                   // newctu-warning{{TRUE}} ctu
                                                            // newctu-warning@-1{{UNKNOWN}} stu
                                                            // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(myns::embed_ns::fens(2) == -1);       // newctu-warning{{TRUE}} ctu
                                                            // newctu-warning@-1{{UNKNOWN}} stu
                                                            // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(mycls().fcl(1) == 6);                 // newctu-warning{{TRUE}} ctu
                                                            // newctu-warning@-1{{UNKNOWN}} stu
                                                            // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(mycls::fscl(1) == 7);                 // newctu-warning{{TRUE}} ctu
                                                            // newctu-warning@-1{{UNKNOWN}} stu
                                                            // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(myns::embed_cls().fecl(1) == -6);     // newctu-warning{{TRUE}} ctu
                                                            // newctu-warning@-1{{UNKNOWN}} stu
                                                            // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(mycls::embed_cls2().fecl2(0) == -11); // newctu-warning{{TRUE}} ctu
                                                            // newctu-warning@-1{{UNKNOWN}} stu
                                                            // oldctu-warning@-2{{TRUE}}

  clang_analyzer_eval(chns::chf1(4) == 12); // newctu-warning{{TRUE}} ctu
                                            // newctu-warning@-1{{UNKNOWN}} stu
                                            // oldctu-warning@-2{{TRUE}}
  clang_analyzer_eval(fun_using_anon_struct(8) == 8); // newctu-warning{{TRUE}} ctu
                                                      // newctu-warning@-1{{UNKNOWN}} stu
                                                      // oldctu-warning@-2{{TRUE}}

  clang_analyzer_eval(other_macro_diag(1) == 1); // newctu-warning{{TRUE}} ctu
                                                 // newctu-warning@-1{{UNKNOWN}} stu
                                                 // oldctu-warning@-2{{TRUE}}
  // newctu-warning@Inputs/ctu-other.cpp:93{{REACHABLE}}
  // oldctu-warning@Inputs/ctu-other.cpp:93{{REACHABLE}}
  MACRODIAG(); // newctu-warning{{REACHABLE}}
               // oldctu-warning@-1{{REACHABLE}}

  // FIXME we should report an UNKNOWN as well for all external variables!
  clang_analyzer_eval(extInt == 2); // newctu-warning{{TRUE}}
                                    // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(intns::extInt == 3); // newctu-warning{{TRUE}}
                                           // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(extS.a == 4); // newctu-warning{{TRUE}}
                                    // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(extNonConstS.a == 4); // newctu-warning{{UNKNOWN}}
                                            // oldctu-warning@-1{{UNKNOWN}}
  // Do not import non-trivial classes' initializers.
  clang_analyzer_eval(extNTS.a == 4); // newctu-warning{{UNKNOWN}}
                                      // oldctu-warning@-1{{UNKNOWN}}
  clang_analyzer_eval(extHere == 6); // newctu-warning{{TRUE}}
                                     // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(A::a == 3); // newctu-warning{{TRUE}}
                                  // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(extSC.a == 8); // newctu-warning{{TRUE}}
                                     // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(ST::sc.a == 2); // newctu-warning{{TRUE}}
                                      // oldctu-warning@-1{{TRUE}}
  // clang_analyzer_eval(extSCN.scn.a == 9); // TODO
  clang_analyzer_eval(extSubSCN.a == 1); // newctu-warning{{TRUE}}
                                         // oldctu-warning@-1{{TRUE}}
  // clang_analyzer_eval(extSCC.a == 7); // TODO
  clang_analyzer_eval(extU.a == 4); // newctu-warning{{TRUE}}
                                    // oldctu-warning@-1{{TRUE}}
  clang_analyzer_eval(TestAnonUnionUSR::Test == 5); // newctu-warning{{TRUE}}
                                                    // oldctu-warning@-1{{TRUE}}

  clang_analyzer_eval(testImportOfIncompleteDefaultParmDuringImport(9) == 9);
  // newctu-warning@-1{{TRUE}} ctu
  // newctu-warning@-2{{UNKNOWN}} stu
  // oldctu-warning@-3{{TRUE}}

  clang_analyzer_eval(testImportOfDelegateConstructor(10) == 10);
  // newctu-warning@-1{{TRUE}} ctu
  // newctu-warning@-2{{UNKNOWN}} stu
  // oldctu-warning@-3{{TRUE}}
}
