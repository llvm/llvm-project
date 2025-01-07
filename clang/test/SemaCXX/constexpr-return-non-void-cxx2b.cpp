// RUN: %clang_cc1 -std=c++23 -fsyntax-only -Wimplicit-fallthrough -Wconsumed -verify %s

constexpr int f() { } // expected-warning {{non-void function does not return a value}}
static_assert(__is_same(decltype([] constexpr -> int { }( )), int)); // expected-warning {{non-void lambda does not return a value}}

consteval int g() { } // expected-warning {{non-void function does not return a value}}
static_assert(__is_same(decltype([] consteval -> int { }( )), int)); // expected-warning {{non-void lambda does not return a value}}

namespace GH116485 {
int h() {
    if consteval { }
} // expected-warning {{non-void function does not return a value}}

void i(int x) {
  if consteval {
  }
  switch (x) {
  case 1:
    i(1);
  case 2: // expected-warning {{unannotated fall-through between switch labels}} \
          // expected-note {{insert 'break;' to avoid fall-through}}
    break;
  }
}

constexpr bool j()  {
    if !consteval { return true; }
} // expected-warning {{non-void function does not return a value in all control paths}} \
  // expected-note {{control reached end of constexpr function}}

bool k = j();
constinit bool l = j(); // expected-error {{variable does not have a constant initializer}} \
                        // expected-note {{required by 'constinit' specifier here}} \
                        // expected-note {{in call to 'j()'}}

}

namespace GH117385 {
void f() {
  if consteval {
  }
}
}
