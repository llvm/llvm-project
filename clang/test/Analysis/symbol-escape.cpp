// RUN: %clang_analyze_cc1 \
// RUN:  -analyzer-checker=core,cplusplus.NewDeleteLeaks \
// RUN:  -verify %s

// expected-no-diagnostics: Whenever we cannot evaluate an operation we escape
//                          the operands. After the evaluation it would be an
//                          Unknown value and the tracking would be lost.

typedef unsigned __INTPTR_TYPE__ uintptr_t;

class C {};

C *simple_escape_in_bitwise_op(C *Foo) {
  C *Bar = new C();
  Bar = reinterpret_cast<C *>(reinterpret_cast<uintptr_t>(Bar) & 0x1);
  (void)Bar;
  // no-warning: "Potential leak of memory pointed to by 'Bar'" was here.

  return Bar;
}

C **indirect_escape_in_bitwise_op() {
  C *Qux = new C();
  C **Baz = &Qux;
  Baz = reinterpret_cast<C **>(reinterpret_cast<uintptr_t>(Baz) | 0x1);
  Baz = reinterpret_cast<C **>(reinterpret_cast<uintptr_t>(Baz) &
		               ~static_cast<uintptr_t>(0x1));
  // no-warning: "Potential leak of memory pointed to by 'Qux'" was here.

  delete *Baz;
  return Baz;
}

