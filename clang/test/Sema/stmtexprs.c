// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only -Wno-gnu-statement-expression

int stmtexpr_fn(void);
void stmtexprs(int i) {
  __builtin_assume( ({ 1; }) ); // no warning about "side effects"
  __builtin_assume( ({ if (i) { (void)0; }; 42; }) ); // no warning about "side effects"
  // expected-warning@+1 {{assumption is ignored because it contains (potential) side-effects}}
  __builtin_assume( ({ if (i) ({ stmtexpr_fn(); }); 1; }) );
}

struct S {
  unsigned b : 3;
};

void test_bitfield_promotion(struct S s) {
  _Static_assert(_Generic(+({ s.b; }), int: 1, unsigned: 2) == 1,
                 "bit-field in statement expression should be promoted");
  _Static_assert(_Generic(+s.b, int: 1, unsigned: 2) == 1,
                 "ordinary bit-field access should be promoted");
  _Static_assert(_Generic(({ s.b; }), int: 1, unsigned: 2) == 2,
                 "bit-field in statement expression without unary + should not be promoted");
  _Static_assert(_Generic(s.b, int: 1, unsigned: 2) == 2,
                 "ordinary bit-field access without unary + should not be promoted");
}
