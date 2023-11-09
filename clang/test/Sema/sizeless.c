// RUN: %clang_cc1 %s -fsyntax-only -verify

struct T { // expected-note {{forward declaration of 'struct T'}}  expected-note {{forward declaration of 'struct T'}}  expected-note {{forward declaration of 'struct T'}}  expected-note {{forward declaration of 'struct T'}}
	int __attribute__((sizeless)) x;
	float y;
};

void f(void) {
  int size_intty[sizeof(int __attribute__((sizeless))) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type 'int __attribute__((sizeless))'}}
  int align_intty[__alignof__(int __attribute__((sizeless))) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type 'int __attribute__((sizeless))'}}
  
  int __attribute__((sizeless)) var1;
  int size_int[sizeof(var1) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type 'int __attribute__((sizeless))'}}
  int align_int[__alignof__(var1) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type 'int __attribute__((sizeless))'}}

  int size_struct[sizeof(struct T) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type 'struct T'}}
  int align_struct[__alignof__(struct T) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type 'struct T'}}
  
  struct T var2;
  int size_structty[sizeof(var2) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type 'struct T'}}
  int align_structty[__alignof__(var2) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type 'struct T'}}
}
