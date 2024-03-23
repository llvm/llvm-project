// RUN: %clang_analyze_cc1 -std=c++2b -verify %s \
// RUN:   -analyzer-checker=core,debug.ExprInspection

template <typename T> void clang_analyzer_dump(T);

struct Adder {
  int data;
  static int operator()(int x, int y) {
    clang_analyzer_dump(x); // expected-warning {{1}}
    clang_analyzer_dump(y); // expected-warning {{2}}
    return x + y;
  }
};

void static_operator_call_inlines() {
  Adder s{10};
  clang_analyzer_dump(s(1, 2)); // expected-warning {{3}}
}

struct DataWithCtor {
  int x;
  int y;
  DataWithCtor(int parm) : x(parm + 10), y(parm + 20) {
    clang_analyzer_dump(this); // expected-warning {{&v}}
  }
};

struct StaticSubscript {
  static void operator[](DataWithCtor v) {
    clang_analyzer_dump(v.x); // expected-warning {{20}}
    clang_analyzer_dump(v.y); // expected-warning {{30}}
  }
};

void top() {
  StaticSubscript s;
  s[DataWithCtor{10}];
}
