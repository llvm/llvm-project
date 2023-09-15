// RUN: %clang_analyze_cc1 -analyzer-checker=core %s \
// RUN:    -triple x86_64-pc-linux-gnu -verify

void clang_analyzer_eval(int);

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
  if (p == (int *)0x1b) {
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
  if (p == (int *)0x1b) {
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

void test_simplified_before_cast_add(long t1) {
  long long t2 = t1 + 3;
  if (!t2) {
    int *p = (int *) t2;
    clang_analyzer_eval(p == 0); // expected-warning{{TRUE}}
  }
}

void test_simplified_before_cast_sub(long t1) {
  long long t2 = t1 - 3;
  if (!t2) {
    int *p = (int *) t2;
    clang_analyzer_eval(p == 0); // expected-warning{{TRUE}}
  }
}

void test_simplified_before_cast_mul(long t1) {
  long long t2 = t1 * 3;
  if (!t2) {
    int *p = (int *) t2;
    clang_analyzer_eval(p == 0); // expected-warning{{TRUE}}
  }
}
