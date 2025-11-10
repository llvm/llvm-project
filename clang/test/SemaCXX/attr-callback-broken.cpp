// RUN: %clang_cc1 %s -std=c++23 -verify -fsyntax-only

class C_in_class {
#define HAS_THIS
#include "../Sema/attr-callback-broken.c"
#undef HAS_THIS
};

class ExplicitParameterObject {
  __attribute__((callback(2, 0))) void explicit_this_idx(this ExplicitParameterObject* self, void (*callback)(ExplicitParameterObject*));           // expected-error {{'callback' argument at position 2 references unavailable implicit 'this'}}
  __attribute__((callback(2, this))) void explicit_this_identifier(this ExplicitParameterObject* self, void (*callback)(ExplicitParameterObject*)); // expected-error {{'callback' argument at position 2 references unavailable implicit 'this'}}
};
