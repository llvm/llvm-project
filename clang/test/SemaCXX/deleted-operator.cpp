// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct PR10757 {
  bool operator~() = delete; // expected-note {{explicitly deleted}}
  bool operator==(const PR10757&) = delete; // expected-note {{explicitly deleted}}
  operator float();
};
int PR10757f() {
  PR10757 a1;
  // FIXME: We get a ridiculous number of "built-in candidate" notes here...
  if(~a1) {} // expected-error {{overload resolution selected deleted operator}} expected-note 8 {{built-in candidate}}
  if(a1==a1) {} // expected-error {{overload resolution selected deleted operator}} expected-note 121 {{built-in candidate}}
}

struct DelOpDel {
  // FIXME: In MS ABI, we error twice below.
  virtual ~DelOpDel() {} // expected-error 1-2 {{attempt to use a deleted function}}
  void operator delete(void*) = delete; // expected-note 1-2 {{deleted here}}
};
