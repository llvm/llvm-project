// RUN: %clang_cc1 -std=c++20 -fblocks -fsyntax-only -verify %s

void test_lambdas() {
  (void) [] () -> void { return {}; }; // expected-error {{void lambda should not return a value}}
  (void) [] () -> void { return {1}; }; // expected-error {{void lambda should not return a value}}
  (void) [] () -> void { return {1, 2}; }; // expected-error {{void lambda should not return a value}}
  (void) [] () -> void { return 42; }; // expected-error {{void lambda should not return a value}}
  (void) [] () -> void { return double(32); }; // expected-error {{void lambda should not return a value}}
  
  // Qualtype  on void  Lambda return
  (void) [] () -> const void { return {1}; }; // expected-error {{void lambda should not return a value}}
  (void) [] () -> volatile void { // expected-warning {{volatile-qualified return type 'volatile void' is deprecated}}
    return {1, 2}; // expected-error {{void lambda should not return a value}}
  };

  (void) [] () -> void { return ({}); };
  (void) [] () -> void { return void{}; };
  (void) [] () -> void { return void(); };
}

void test_blocks() {
  (void) ^ void { return {}; }; // expected-error {{void block should not return a value}}
  (void) ^ void { return {1}; }; // expected-error {{void block should not return a value}}
  (void) ^ void { return {1, 2}; }; // expected-error {{void block should not return a value}}
  (void) ^ void { return 42; }; // expected-error {{void block should not return a value}}
  
  // Qualtype on void Block return
  (void) ^ const void { return {1}; }; // expected-error {{void block should not return a value}}
}
