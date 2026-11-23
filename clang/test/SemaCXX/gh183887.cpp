// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace GH183887 {
enum E1 explicit(E1()); // expected-error {{ISO C++ forbids forward references to 'enum' types}} \
                        // expected-error {{invalid use of incomplete type 'E1'}} \
                        // expected-error {{'explicit' can only appear on non-static member functions}} \
                        // expected-note {{forward declaration of 'GH183887::E1'}}
}
