// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fsyntax-only -verify %s
// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fsyntax-only -verify %s -frecovery-ast -frecovery-ast-type

// This checks that when a block is discarded, the enclosing lambda’s
// unexpanded parameter pack flag is reset to what it was before the
// block is parsed so we don't crash when trying to diagnose unexpanded
// parameter packs in the lambda.

template <typename ...Ts>
void gh109148() {
  (^Ts); // expected-error {{expected expression}} expected-error {{unexpanded parameter pack 'Ts'}}

  [] {
    (^Ts); // expected-error {{expected expression}}
    ^Ts;   // expected-error {{expected expression}}
    ^(Ts); // expected-error {{expected expression}}
    ^ Ts); // expected-error {{expected expression}}
  };

  ([] {
    (^Ts); // expected-error {{expected expression}}
    ^Ts;   // expected-error {{expected expression}}
    ^(Ts); // expected-error {{expected expression}}
    ^ Ts); // expected-error {{expected expression}}
  }, ...); // expected-error {{pack expansion does not contain any unexpanded parameter packs}}

  [] { // expected-error {{unexpanded parameter pack 'Ts'}}
    ^ (Ts) {};
  };

  [] { // expected-error {{unexpanded parameter pack 'Ts'}}
    (void) ^ { Ts x; };
  };

  [] { // expected-error {{unexpanded parameter pack 'Ts'}}
    Ts s;
    (^Ts); // expected-error {{expected expression}}
  };

  ([] {
    Ts s;
    (^Ts); // expected-error {{expected expression}}
  }, ...);

  [] { // expected-error {{unexpanded parameter pack 'Ts'}}
    ^ { Ts s; return not_defined; }; // expected-error {{use of undeclared identifier 'not_defined'}}
  };
}
