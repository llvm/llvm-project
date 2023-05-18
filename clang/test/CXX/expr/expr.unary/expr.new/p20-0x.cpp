// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -fexceptions %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++14 -fexceptions %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z -fexceptions %s
typedef __SIZE_TYPE__ size_t;

namespace std { enum class align_val_t : size_t {}; }

struct S {
  // Placement allocation function:
  static void* operator new(size_t, size_t);
  // Usual (non-placement) deallocation function:
  static void operator delete(void*, size_t); // expected-note{{declared here}}
};

void testS() {
  S* p = new (0) S; // expected-error{{'new' expression with placement arguments refers to non-placement 'operator delete'}}
}

struct T {
  // Placement allocation function:
  static void* operator new(size_t, size_t);
  // Usual (non-placement) deallocation function:
  static void operator delete(void*);
  // Placement deallocation function:
  static void operator delete(void*, size_t);
};

void testT() {
  T* p = new (0) T; // ok
}

#if __cplusplus > 201402L
struct U {
  // Placement allocation function:
  static void* operator new(size_t, size_t, std::align_val_t);
  // Placement deallocation function:
  static void operator delete(void*, size_t, std::align_val_t); // expected-note{{declared here}}
};

void testU() {
  U* p = new (0, std::align_val_t(0)) U; // expected-error{{'new' expression with placement arguments refers to non-placement 'operator delete'}}
}

struct V {
  // Placement allocation function:
  static void* operator new(size_t, size_t, std::align_val_t);
  // Usual (non-placement) deallocation function:
  static void operator delete(void*, std::align_val_t);
  // Placement deallocation function:
  static void operator delete(void*, size_t, std::align_val_t);
};

void testV() {
  V* p = new (0, std::align_val_t(0)) V;
}

struct W {
  // Placement allocation function:
  static void* operator new(size_t, size_t, std::align_val_t);
  // Usual (non-placement) deallocation functions:
  static void operator delete(void*);
  static void operator delete(void*, size_t, std::align_val_t); // expected-note {{declared here}}
};

void testW() {
  W* p = new (0, std::align_val_t(0)) W; // expected-error{{'new' expression with placement arguments refers to non-placement 'operator delete'}}
}
#endif
