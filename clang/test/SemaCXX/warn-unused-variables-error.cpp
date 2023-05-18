// RUN: %clang_cc1 -fsyntax-only -Wunused-variable -verify %s

namespace PR6948 {
  template<typename T> class X; // expected-note{{template is declared here}}
  
  void f() {
    X<char> str (read_from_file()); // expected-error{{use of undeclared identifier 'read_from_file'}} \
                                       expected-error{{implicit instantiation of undefined template 'PR6948::X<char>'}}
  }
}
