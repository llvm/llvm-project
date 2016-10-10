// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -fexceptions %s
typedef __SIZE_TYPE__ size_t;

struct S {
  // Placement allocation function:
  static void* operator new(size_t, size_t);
  // Usual (non-placement) deallocation function:
  static void operator delete(void*, size_t); // expected-note{{declared here}}
};

void testS() {
  S* p = new (0) S;	// expected-error{{'new' expression with placement arguments refers to non-placement 'operator delete'}}
}
