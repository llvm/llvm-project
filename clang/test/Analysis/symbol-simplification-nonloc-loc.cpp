// RUN: %clang_analyze_cc1 -analyzer-checker=core %s \
// RUN:    -triple x86_64-pc-linux-gnu -verify

#define BINOP(OP) [](auto x, auto y) { return x OP y; }

template <typename BinOp>
void nonloc_OP_loc(int *p, BinOp op) {
  long p_as_integer = (long)p;
  if (op(12, p_as_integer) != 11)
    return;

  // Perfectly constrain 'p', thus 'p_as_integer', and trigger a simplification
  // of the previously recorded constraint.
  if (p) {
    // no-crash
  }
  if (p == (int *)0x404) {
    // no-crash
  }
}

// Same as before, but the operands are swapped.
template <typename BinOp>
void loc_OP_nonloc(int *p, BinOp op) {
  long p_as_integer = (long)p;
  if (op(p_as_integer, 12) != 11)
    return;

  if (p) {
    // no-crash
  }
  if (p == (int *)0x404) {
    // no-crash
  }
}

void instantiate_tests_for_nonloc_OP_loc(int *p) {
  // Multiplicative and additive operators:
  nonloc_OP_loc(p, BINOP(*));
  nonloc_OP_loc(p, BINOP(/)); // no-crash
  nonloc_OP_loc(p, BINOP(%)); // no-crash
  nonloc_OP_loc(p, BINOP(+));
  nonloc_OP_loc(p, BINOP(-)); // no-crash

  // Bitwise operators:
  // expected-warning@+2 {{The result of the left shift is undefined due to shifting by '1028', which is greater or equal to the width of type 'int' [core.UndefinedBinaryOperatorResult]}}
  // expected-warning@+2 {{The result of the right shift is undefined due to shifting by '1028', which is greater or equal to the width of type 'int' [core.UndefinedBinaryOperatorResult]}}
  nonloc_OP_loc(p, BINOP(<<)); // no-crash
  nonloc_OP_loc(p, BINOP(>>)); // no-crash
  nonloc_OP_loc(p, BINOP(&));
  nonloc_OP_loc(p, BINOP(^));
  nonloc_OP_loc(p, BINOP(|));
}

void instantiate_tests_for_loc_OP_nonloc(int *p) {
  // Multiplicative and additive operators:
  loc_OP_nonloc(p, BINOP(*));
  loc_OP_nonloc(p, BINOP(/));
  loc_OP_nonloc(p, BINOP(%));
  loc_OP_nonloc(p, BINOP(+));
  loc_OP_nonloc(p, BINOP(-));

  // Bitwise operators:
  loc_OP_nonloc(p, BINOP(<<));
  loc_OP_nonloc(p, BINOP(>>));
  loc_OP_nonloc(p, BINOP(&));
  loc_OP_nonloc(p, BINOP(^));
  loc_OP_nonloc(p, BINOP(|));
}

// from: nullptr.cpp
void zoo1backwards() {
  char **p = nullptr;
  // expected-warning@+1 {{Dereference of null pointer [core.NullDereference]}}
  *(0 + p) = nullptr;  // warn
  **(0 + p) = 'a';     // no-warning: this should be unreachable
}
