// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify -std=c99 -Dbool=_Bool -Dtrue=1 -Dfalse=0 %s
// RUN: %clang_analyze_cc1 -analyzer-checker=debug.ExprInspection -verify -std=c11 -Dbool=_Bool -Dtrue=1 -Dfalse=0 %s
extern void clang_analyzer_eval(bool);

void test__Bool_value(void) {
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

void test__Bool_increment(void) {
  {
    bool b = true;
    b++;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = false;
    b++;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = true;
    ++b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = false;
    ++b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = 0;
    ++b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = 10;
    ++b;
    ++b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = -10;
    ++b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = -1;
    ++b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }
}

void test__Bool_decrement(void) {
  {
    bool b = true;
    b--;
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
  }

  {
    bool b = false;
    b--;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = true;
    --b;
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
  }

  {
    bool b = false;
    --b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = 0;
    --b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = 10;
    --b;
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
    --b;
    clang_analyzer_eval(b == 1); // expected-warning{{TRUE}}
  }

  {
    bool b = -10;
    --b;
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
  }

  {
    bool b = 1;
    --b;
    clang_analyzer_eval(b == 0); // expected-warning{{TRUE}}
  }
}
