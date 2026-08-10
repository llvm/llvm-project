// RUN: %clang_cc1 -verify -x c++ -std=c++98 %s
// RUN: not %clang_cc1 -fdiagnostics-parseable-fixits -x c++ -std=c++98 %s 2>&1 | FileCheck %s

struct S {
  int n;
};

struct T {
  T();
  T(S, S);
  int n;
};

struct U {
  ~U();
  int n;
};

struct V {
  ~V();
};

struct W : V {
};

struct X : U {
};

int F1();
S F2();

namespace N {
  void test() {
    // CHECK: fix-it:"{{.*}}":{35:9-35:11}:" = {}"
    S s1(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{39:9-39:10}:";"
    // CHECK: fix-it:"{{.*}}":{40:7-40:9}:" = {}"
    S s2, // expected-note {{change this ',' to a ';' to call 'F2'}}
    F2(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{44:9-44:11}:""
    // CHECK: fix-it:"{{.*}}":{45:9-45:11}:""
    T t1(), // expected-warning {{function declaration}} expected-note {{remove parentheses}}
      t2(); // expected-warning {{function declaration}} expected-note {{remove parentheses}}

    // Suggest parentheses only around the first argument.
    // CHECK: fix-it:"{{.*}}":{50:10-50:10}:"("
    // CHECK: fix-it:"{{.*}}":{50:13-50:13}:")"
    T t3(S(), S()); // expected-warning {{disambiguated as a function declaration}} expected-note {{add a pair of parentheses}}

    // Check fixit position for pathological case
    // CHECK: fix-it:"{{.*}}":{56:11-56:11}:"("
    // CHECK: fix-it:"{{.*}}":{56:20-56:20}:")"
    float k[1];
    int l(int(k[0])); // expected-warning {{disambiguated as a function declaration}} expected-note {{add a pair of parentheses}}

    // Don't emit warning and fixit because this must be a function declaration due to void return type.
    typedef void VO;
    VO m(int (*p)[4]);

    // Don't emit warning and fixit because direct initializer is not permitted here.
    if (int n(int())){} // expected-error {{function type is not allowed here}}

    // CHECK: fix-it:"{{.*}}":{66:8-66:10}:" = {}"
    U u(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{69:8-69:10}:""
    V v(); // expected-warning {{function declaration}} expected-note {{remove parentheses}}

    // CHECK: fix-it:"{{.*}}":{72:8-72:10}:""
    W w(); // expected-warning {{function declaration}} expected-note {{remove parentheses}}

    // TODO: Removing the parens here would not initialize U::n.
    // Maybe suggest an " = X()" initializer for this case?
    // Maybe suggest removing the parens anyway?
    X x(); // expected-warning {{function declaration}}

    // CHECK: fix-it:"{{.*}}":{80:11-80:13}:" = 0"
    int n1(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{84:11-84:12}:";"
    // CHECK: fix-it:"{{.*}}":{85:7-85:9}:" = 0"
    int n2, // expected-note {{change this ',' to a ';' to call 'F1'}}
    F1(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{88:13-88:15}:" = 0.0"
    double d(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    typedef void *Ptr;

    // CHECK: fix-it:"{{.*}}":{93:10-93:12}:" = 0"
    Ptr p(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

#define NULL 0
    // CHECK: fix-it:"{{.*}}":{97:10-97:12}:" = NULL"
    Ptr p(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{100:11-100:13}:" = false"
    bool b(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{103:11-103:13}:" = '\\0'"
    char c(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}

    // CHECK: fix-it:"{{.*}}":{106:15-106:17}:" = L'\\0'"
    wchar_t wc(); // expected-warning {{function declaration}} expected-note {{replace parentheses with an initializer}}
  }
}

namespace RedundantParens {
struct Y {
  Y();
  Y(int);
  ~Y();
};
int n;

void test() {
  // CHECK: add a variable name
  // CHECK: fix-it:"{{.*}}":{[[@LINE+7]]:4-[[@LINE+7]]:4}:" varname"
  // CHECK: add enclosing parentheses
  // CHECK: fix-it:"{{.*}}":{[[@LINE+5]]:3-[[@LINE+5]]:3}:"("
  // CHECK: fix-it:"{{.*}}":{[[@LINE+4]]:7-[[@LINE+4]]:7}:")"
  // CHECK: remove parentheses
  // CHECK: fix-it:"{{.*}}":{[[@LINE+2]]:4-[[@LINE+2]]:5}:" "
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:6-[[@LINE+1]]:7}:""
  Y(n); // expected-warning {{declaration of variable named 'n'}} expected-note 3{{}}
}
}
