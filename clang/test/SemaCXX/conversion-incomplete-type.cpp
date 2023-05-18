// RUN: %clang_cc1 -fsyntax-only -verify %s

struct string {};

class StringPiece;  // expected-note {{forward declaration of 'StringPiece'}} \
                    // expected-note {{forward declaration of 'StringPiece'}}

struct Test {
  void expectStringPiece(const StringPiece& blah) {};  // expected-note {{passing argument to parameter 'blah' here}}
  
  void test(const string& s) { 
    expectStringPiece(s);  // expected-error {{no viable conversion from 'const string' to incomplete type 'const StringPiece'}}
  }
};

struct TestStatic {
  static void expectStringPiece(const StringPiece& blah) {};  // expected-note {{passing argument to parameter 'blah' here}}
  
  static void test(const string& s) { 
    expectStringPiece(s);  // expected-error {{no viable conversion from 'const string' to incomplete type 'const StringPiece'}}
  }
};

