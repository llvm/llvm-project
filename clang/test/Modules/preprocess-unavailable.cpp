// RUN: %clang_cc1 -x c++-module-map %s -fmodule-name=a -verify -std=c++98
module a {
  module b {
    requires cplusplus11
  }
}
#pragma clang module contents
// expected-error@3 {{module 'a.b' requires feature 'cplusplus11'}}
#pragma clang module begin a.b // expected-note {{entering module 'a' due to this pragma}}
int f();
int g() { f(); }
#pragma clang module end // expected-error {{no matching '#pragma clang module begin'}}
