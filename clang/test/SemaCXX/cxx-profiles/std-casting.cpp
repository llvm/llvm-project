// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 \
// RUN:   -fexperimental-cxx-profiles -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 \
// RUN:   -fexperimental-cxx-profiles -fcxx-profiles=std::casting -DCMDLINE \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -std=c++20 \
// RUN:   -fexperimental-cxx-profiles -DCMDLINE -fsyntax-only \
// RUN:   -verify=not-enforced %s

// not-enforced-no-diagnostics

#ifndef CMDLINE
[[profiles::enforce(std::casting)]];
#endif

namespace std {
enum class byte : unsigned char {};
}

struct B {
  virtual ~B();
};
struct D : B {};
struct NB {};
struct ND : NB {};
using FloatPtr = float *;

void reinterpret(int *ip, double *dp, unsigned long ul) {
  (void)reinterpret_cast<float *>(ip);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'float *'}}
  // expected-note@-2 {{use 'std::bit_cast' to reinterpret the object representation}}
  (void)reinterpret_cast<float *>(dp);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'double *' to 'float *'}}
  (void)reinterpret_cast<char *>(ip);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'char *'}}
  (void)reinterpret_cast<std::byte *>(ip);
  (void)reinterpret_cast<const std::byte &>(*ip);
  (void)reinterpret_cast<__UINTPTR_TYPE__>(ip);
  (void)reinterpret_cast<int *>(ul);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'unsigned long' to 'int *'}}

  (void)(float *)ip;
  // expected-error@-1 {{std::casting forbids a C-style cast that performs a reinterpret_cast from 'int *' to 'float *'}}
  // expected-note@-2 {{use 'reinterpret_cast' to make the conversion explicit}}
  // expected-note@-3 {{use 'std::bit_cast' to reinterpret the object representation}}
  (void)FloatPtr(ip);
  // expected-error@-1 {{std::casting forbids a functional-style cast that performs a reinterpret_cast from 'int *' to 'FloatPtr' (aka 'float *')}}
  // expected-note@-2 {{use 'reinterpret_cast' to make the conversion explicit}}
  // expected-note@-3 {{use 'std::bit_cast' to reinterpret the object representation}}
}

void constness(const int *cip, const int &cr, volatile int *vip,
               const char *cc) {
  (void)const_cast<int *>(cip);
  // expected-error@-1 {{std::casting forbids const_cast that casts away constness from 'const int *' to 'int *'}}
  (void)const_cast<int &>(cr);
  // expected-error@-1 {{std::casting forbids const_cast that casts away constness from 'const int' to 'int &'}}
  (void)const_cast<int *>(vip);
  (void)const_cast<const int *>(cip);

  (void)(int *)cip;
  // expected-error@-1 {{std::casting forbids a C-style cast that casts away constness from 'const int *' to 'int *'}}
  // expected-note@-2 {{use 'const_cast' to make the conversion explicit}}
  (void)(int *)cc;
  // expected-error@-1 {{std::casting forbids a C-style cast that performs a reinterpret_cast from 'const char *' to 'int *'}}
  // expected-error@-2 {{std::casting forbids a C-style cast that casts away constness from 'const char *' to 'int *'}}
}

void downcast(B *b, B &br, NB *nb) {
  (void)static_cast<D *>(b);
  // expected-error@-1 {{std::casting forbids static_cast from base class 'B' to derived class 'D'}}
  // expected-note@-2 {{use 'dynamic_cast' to perform a checked conversion}}
  (void)static_cast<D &>(br);
  // expected-error@-1 {{std::casting forbids static_cast from base class 'B' to derived class 'D'}}
  // expected-note@-2 {{use 'dynamic_cast' to perform a checked conversion}}
  (void)static_cast<ND *>(nb);
  // expected-error@-1 {{std::casting forbids static_cast from base class 'NB' to derived class 'ND'}}
  (void)dynamic_cast<D *>(b);
  (void)static_cast<B *>(static_cast<D *>(nullptr));

  (void)(D *)b;
  // expected-error@-1 {{std::casting forbids a C-style cast from base class 'B' to derived class 'D'}}
  // expected-note@-2 {{use 'static_cast' to make the conversion explicit}}
  // expected-note@-3 {{use 'dynamic_cast' to perform a checked conversion}}
}

void from_void(void *vp, const void *cvp) {
  (void)static_cast<int *>(vp);
  // expected-error@-1 {{std::casting forbids static_cast from 'void *' to 'int *'}}
  (void)static_cast<std::byte *>(vp);
  (void)static_cast<const void *>(vp);
  (void)static_cast<const int *>(cvp);
  // expected-error@-1 {{std::casting forbids static_cast from 'const void *' to 'const int *'}}
}

void allowed(float f, int i, long l, B *b) {
  (void)static_cast<int>(f);
  (void)(int)f;
  (void)int(f);
  (void)(long)i;
  (void)static_cast<B *>(b);
  (void)(void)b;
}

void unevaluated(int *ip) {
  (void)sizeof(reinterpret_cast<float *>(ip));
  decltype(reinterpret_cast<float *>(ip)) fp = nullptr;
  (void)fp;
}

template <typename T> float *dependent(T *p) {
  return reinterpret_cast<float *>(p);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'float *'}}
  // expected-note@-2 {{use 'std::bit_cast' to reinterpret the object representation}}
  // expected-note@#dependent-int {{in instantiation of function template specialization 'dependent<int>' requested here}}
}

template <typename T> float *non_dependent(T *, int *ip) {
  return reinterpret_cast<float *>(ip);
  // expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'float *'}}
  // expected-note@-2 {{use 'std::bit_cast' to reinterpret the object representation}}
}

void use_templates(int *ip, long *lp) {
  // A violation in a template is reported once, not once per instantiation.
  (void)dependent(ip); // #dependent-int
  (void)dependent(lp);
  (void)non_dependent(ip, ip);
  (void)non_dependent(lp, ip);
}

float *global = reinterpret_cast<float *>(static_cast<int *>(nullptr));
// expected-error@-1 {{std::casting forbids reinterpret_cast from 'int *' to 'float *'}}
// expected-note@-2 {{use 'std::bit_cast' to reinterpret the object representation}}
