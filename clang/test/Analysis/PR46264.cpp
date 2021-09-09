// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// rdar://problem/64202361

struct A {
  int a;
  struct {
    struct {
      int b;
      union {
        int c;
      };
    };
  };
};

int testCrash() {
  int *x = 0;
  int A::*ap = &A::a;

  if (ap)      // no crash
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}

  return 10;
}

int testIndirectCrash() {
  int *x = 0;
  int A::*cp = &A::c;

  if (cp)      // no crash
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}

  return 10;
}

// PR46264
// This case shall not crash with an assertion failure about void* dereferening.
namespace ns1 {
namespace a {
class b {
public:
  typedef int b::*c;
  operator c() { return d ? &b::d : 0; }
  int d;
};
} // namespace a
using a::b;
class e {
  void f();
  void g();
  b h;
};
void e::f() {
  e *i;
  if (h)
    i->g(); // expected-warning{{Called C++ object pointer is uninitialized}}
}
} // namespace ns1
