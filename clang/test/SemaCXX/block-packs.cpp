// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fsyntax-only -verify -Wno-unused %s
// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fsyntax-only -verify -Wno-unused %s -frecovery-ast -frecovery-ast-type

// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fsyntax-only -verify -Wno-unused -fexperimental-new-constant-interpreter %s
// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -fsyntax-only -verify -Wno-unused -frecovery-ast -frecovery-ast-type -fexperimental-new-constant-interpreter %s

template <typename ...Ts>
void f() {
  ((^ { Ts t; }), ...);
  ((^ (Ts t) {}), ...);
  ((^ Ts () {}), ...);

  ^ { Ts t; }; // expected-error {{unexpanded parameter pack 'Ts'}}
  ^ (Ts t) {}; // expected-error {{unexpanded parameter pack 'Ts'}}
  ^ Ts () {};  // expected-error {{unexpanded parameter pack 'Ts'}}
}

template <typename ...Ts>
void gh109148() {
  (^Ts); // expected-error {{expected expression}}

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
    ^ { Ts x; };
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

void g() {
  f<>();
  f<int>();
  f<long, float>();

  gh109148<>();
  gh109148<int>();
  gh109148<long, float>();
}
