// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-inlining=destructors -verify -std=c++11 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-inlining=destructors -verify -std=c++17 %s

void clang_analyzer_eval(bool);

struct InlineDtor {
  static int cnt;
  static int dtorCalled;
  ~InlineDtor() {
    ++dtorCalled;
  }
};

int InlineDtor::cnt = 0;
int InlineDtor::dtorCalled = 0;

void testUnionDtor() {
  static int unionDtorCalled;
  InlineDtor::cnt = 0;
  InlineDtor::dtorCalled = 0;
  unionDtorCalled = 0;
  {
      union UnionDtor {
          InlineDtor kind1;
          char kind2;
          ~UnionDtor() { unionDtorCalled++; }
      };
      UnionDtor u1{.kind1{}};
      UnionDtor u2{.kind2{}};
      auto u3 = new UnionDtor{.kind1{}};
      auto u4 = new UnionDtor{.kind2{}};
      delete u3;
      delete u4;
  }

  clang_analyzer_eval(unionDtorCalled == 4); // expected-warning {{TRUE}}
  clang_analyzer_eval(InlineDtor::dtorCalled == 0); // expected-warning {{TRUE}}
}
