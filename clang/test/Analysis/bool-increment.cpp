// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify -std=c++98 -Wno-deprecated %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify -std=c++11 -Wno-deprecated %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify -std=c++14 -Wno-deprecated %s

extern void clang_analyzer_eval(bool);

void test_bool_value() {
  {
    bool b = true;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = false;
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
  }

  {
    bool b = -10;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = 10;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = 10;
    b++;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = 0;
    b++;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }
}

void test_bool_increment() {
  {
    bool b = true;
    b++;
    clang_analyzer_eval(b); // expected-warning{{TRUE}}
  }

  {
    bool b = false;
    b++;
    clang_analyzer_eval(b); // expected-warning{{TRUE}}
  }

  {
    bool b = true;
    ++b;
    clang_analyzer_eval(b); // expected-warning{{TRUE}}
  }

  {
    bool b = false;
    ++b;
    clang_analyzer_eval(b); // expected-warning{{TRUE}}
  }

  {
    bool b = 0;
    ++b;
    clang_analyzer_eval(b); // expected-warning{{TRUE}}
  }

  {
    bool b = 10;
    ++b;
    ++b;
    clang_analyzer_eval(b); // expected-warning{{TRUE}}
  }

  {
    bool b = -10;
    ++b;
    clang_analyzer_eval(b); // expected-warning{{TRUE}}
  }
}
