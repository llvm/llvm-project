// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

void f(void* p) {
  (void)*p; // expected-error{{indirection not permitted on operand of type 'void *'}}
}

template<class T>
concept deref = requires (T& t) {
      { *t }; // #FAILED_REQ
};

static_assert(deref<void*>);
// expected-error@-1{{static assertion failed}}
// expected-note@-2{{because 'void *' does not satisfy 'deref'}}
// expected-note@#FAILED_REQ{{because '*t' would be invalid: indirection not permitted on operand of type 'void *'}}
