// RUN: %clang_cc1 -fsyntax-only -std=c++20 -ferror-limit 1 -verify %s

template <class>
concept f = requires { 42; };
struct h {
  // The missing semicolon will trigger an error and -ferror-limit=1 will make it fatal
  // We test that we do not crash in such cases (#55401)
  int i = requires { { i } f } // expected-error {{expected ';' at end of declaration list}}
                               // expected-error@* {{too many errors emitted}}
};
