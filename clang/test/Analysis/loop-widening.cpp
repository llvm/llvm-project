// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config widen-loops=true \
// RUN:   -analyzer-config track-conditions=false \
// RUN:   -analyzer-max-loop 2 -analyzer-output=text

namespace pr43102 {
class A {
public:
  void m_fn1();
};
bool g;
void fn1() {
  A a;
  A *b = &a;

  for (;;) { // expected-note{{Loop condition is true.  Entering loop body}}
             // expected-note@-1{{Loop condition is true.  Entering loop body}}
             // expected-note@-2{{Value assigned to 'b'}}
             // no crash during bug report construction

    g = !b;     // expected-note{{Assuming 'b' is null}}
    b->m_fn1(); // expected-warning{{Called C++ object pointer is null}}
                // expected-note@-1{{Called C++ object pointer is null}}
  }
}
} // end of namespace pr43102
