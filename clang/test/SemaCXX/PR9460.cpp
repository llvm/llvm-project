// RUN: %clang_cc1 -fsyntax-only -verify %s

// Don't crash.

template<typename aT>
struct basic_string{
  a; // expected-error {{a type specifier is required}}
  basic_string(aT*);
};

struct runtime_error{
  runtime_error(
basic_string<char> struct{ // expected-error {{cannot combine with previous 'type-name' declaration specifier}}
a(){ // expected-error {{a type specifier is required}}
  runtime_error(0);
}
}
);
};
