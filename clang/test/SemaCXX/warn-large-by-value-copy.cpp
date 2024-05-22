// RUN: %clang_cc1 -verify -fsyntax-only -Wlarge-by-value-copy=100 %s

namespace rdar8548050 {

struct S100 {
    char x[100];
};

struct S101 {
    char x[101];
};

S100 f100(S100 s) { return s; }

S101 f101(S101 s) { return s; } // expected-warning {{return value of 'f101' is a large (101 bytes) pass-by-value object}} \
                                // expected-warning {{'s' is a large (101 bytes) pass-by-value argument}}

void f101_no_param_name(S101) {} // expected-warning {{'' is a large (101 bytes) pass-by-value argument}}

// FIXME: Don't warn when when the return value is subject to (N)RVO.

template <typename T> T foo_template(T);
template <> S101 foo_template(S101) { return S101(); } // expected-warning {{return value of 'foo_template<rdar8548050::S101>' is a large}}
                                                       // expected-warning@-1 {{'' is a large (101 bytes) pass-by-value argument}}

typedef int Arr[200];
void farr(Arr a) { }

struct NonPOD {
  char x[200];
  virtual void m();
};

NonPOD fNonPOD(NonPOD s) { return s; }

template <unsigned size>
struct TS {
    char x[size];
};

template <unsigned size>
void tf(TS<size> ts) {} // expected-warning {{ts' is a large (300 bytes) pass-by-value argument}}

void g() {
    TS<300> ts;
    tf<300>(ts); // expected-note {{instantiation}}
}

}

template<typename T> class DependentPOD {
  enum b { x };
  b foo() { return x; }
};
