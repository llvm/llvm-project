// RUN: %clang_cc1 -triple i686-pc-win32 -std=c++11 -verify %s

struct S {
  virtual ~S() = delete; // expected-note {{'~S' has been explicitly marked deleted here}}
  void operator delete(void*, int);
  void operator delete(void*, double);
} s; // expected-error {{attempt to use a deleted function}}

struct T { // expected-note 2{{virtual destructor requires an unambiguous, accessible 'operator delete'}}
  virtual ~T() = default; // expected-note {{explicitly defaulted function was implicitly deleted here}} expected-warning {{implicitly deleted}} expected-note{{replace 'default'}}
  void operator delete(void*, int);
  void operator delete(void*, double);
} t; // expected-error {{attempt to use a deleted function}}
