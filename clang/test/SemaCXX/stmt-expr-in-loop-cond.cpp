// RUN: %clang_cc1 -std=c++17 -fsyntax-only -Wunreachable-code -verify %s

// Do not crash when a condition variable or an init-statement is initialized
// by a statement-expression containing control flow.  Such an initializer
// finishes the current CFG block, so the builder must not assume the block it
// started with is still current.
// https://github.com/llvm/llvm-project/issues/211976

// expected-no-diagnostics

struct S { explicit operator bool(); };

// A non-trivial destructor exercises the scope/destructor bookkeeping.
struct D { D(); ~D(); explicit operator bool(); };
struct I { I(); ~I(); operator int(); };

void while_loop(int n) {
  while (S s = ({ while (S t{}) {} S{}; }))
    --n;
}

void for_loop(int n) {
  for (; S s = ({ while (S t{}) {} S{}; });)
    --n;
}

void for_loop_with_init_and_inc(int n) {
  for (int i = 0; S s = ({ while (S t{}) {} S{}; }); ++i) {
    if (i > n)
      break;
    continue;
  }
}

void do_loop(int n) {
  do {
    --n;
  } while (({ while (S t{}) {} n; }));
}

void if_condition_variable(int n) {
  if (D d = ({ while (D t{}) {} D{}; }))
    --n;
}

void if_init_statement(int n) {
  if (D d = ({ while (D t{}) {} D{}; }); n)
    --n;
}

void switch_condition_variable(int n) {
  switch (I i = ({ while (D t{}) {} I{}; })) {
  default:
    break;
  }
}

void switch_init_statement(int n) {
  switch (D d = ({ while (D t{}) {} D{}; }); n) {
  default:
    break;
  }
}

// The condition variable is usable in the body, and the nested loop must be
// re-evaluated on every iteration.
void condition_variable_used_in_body(int n) {
  while (D d = ({ while (D t{}) {} D{}; })) {
    if (d)
      continue;
    break;
  }
}
