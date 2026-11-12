// RUN: %clang_cc1 -verify -std=c++2a %s
// expected-no-diagnostics

const unsigned halt = (unsigned)-1;

enum Dir { L, R };
struct Action {
  bool tape;
  Dir dir;
  unsigned next;
};
using State = Action[2];

// An infinite tape!
struct Tape {
  constexpr Tape() = default;
  constexpr ~Tape() {
    if (l) { l->r = nullptr; delete l; }
    if (r) { r->l = nullptr; delete r; }
  }
  constexpr Tape *left() {
    if (!l) { l = new Tape; l->r = this; }
    return l;
  }
  constexpr Tape *right() {
    if (!r) { r = new Tape; r->l = this; }
    return r;
  }
  Tape *l = nullptr;
  bool val = false;
  Tape *r = nullptr;
};

// Run turing machine 'tm' on tape 'tape' from state 'state'. Return number of
// steps taken until halt.
constexpr unsigned run(const State *tm) {
  Tape *tape = new Tape;
  unsigned state = 0;
  unsigned steps = 0;

  for (state = 0; state != halt; ++steps) {
    auto [val, dir, next_state] = tm[state][tape->val];
    tape->val = val;
    tape = (dir == L ? tape->left() : tape->right());
    state = next_state;
  }

  delete tape;
  return steps;
}

// 3-state busy beaver. S(bb3) = 21.
constexpr State bb3[] = {
  { { true, R, 1 }, { true, R, halt } },
  { { true, L, 1 }, { false, R, 2 } },
  { { true, L, 2 }, { true, L, 0 } }
};
static_assert(run(bb3) == 21, "");

// 4-state busy beaver. S(bb4) = 107.
constexpr State bb4[] = {
  { { true, R, 1 }, { true, L, 1 } },
  { { true, L, 0 }, { false, L, 2 } },
  { { true, R, halt }, { true, L, 3 } },
  { { true, R, 3 }, { false, R, 0 } } };
static_assert(run(bb4) == 107, "");
