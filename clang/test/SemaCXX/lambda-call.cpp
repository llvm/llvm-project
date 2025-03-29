// RUN: %clang_cc1 -std=c++23 -verify -fsyntax-only %s

namespace GH96205 {

void f() {
  auto l = [](this auto& self, int) -> void { self("j"); }; // expected-error {{no matching function for call to object of type}} \
                                                            // expected-note {{no known conversion from 'const char[2]' to 'int'}}
  l(3); // expected-note {{requested here}}
}

}
