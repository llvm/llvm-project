// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s

// PR46264
// This case shall not crash with an assertion failure about void* dereferening.
// The crash has been last seen on commit
// `3ed8ebc2f6b8172bed48cc5986d3b7af4cfca1bc` from 24.05.2020.
namespace ns1 {
namespace a {
class b {
public:
  typedef int b::*c;
  operator c() { return d ? &b::d : 0; }
  // expected-note@-1{{'?' condition is true}}
  // expected-note@-2{{Assuming field 'd' is not equal to 0}}
  // expected-note@-3{{Returning value, which participates in a condition later}}
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
  // expected-note@-1{{'i' declared without an initial value}}
  if (h)
    // expected-note@-1{{Taking true branch}}
    // expected-note@-2{{'b::operator int ns1::a::b::*'}}
    // expected-note@-3{{Returning from 'b::operator int ns1::a::b::*'}}
    i->g();
  // expected-note@-1{{Called C++ object pointer is uninitialized}}
  // expected-warning@-2{{Called C++ object pointer is uninitialized}}
}
} // namespace ns1
