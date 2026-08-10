// RUN: %clang_cc1 -verify -std=c++11 %s
// RUN: %clang_cc1 -verify -std=c++11 %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

// A direct proof that constexpr is Turing-complete, once DR1454 is implemented.

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
  constexpr Tape() : l(0), val(false), r(0) {}
  constexpr Tape(const Tape &old, bool write) :
    l(old.l), val(write), r(old.r) {}
  constexpr Tape(const Tape &old, Dir dir) :
    l(dir == L ? old.l ? old.l->l : 0 : &old),
    val(dir == L ? old.l ? old.l->val : false
                 : old.r ? old.r->val : false),
    r(dir == R ? old.r ? old.r->r : 0 : &old) {}
  const Tape *l;
  bool val;
  const Tape *r;
};
constexpr Tape update(const Tape &old, bool write) { return Tape(old, write); }
constexpr Tape move(const Tape &old, Dir dir) { return Tape(old, dir); }

// Run turing machine 'tm' on tape 'tape' from state 'state'. Return number of
// steps taken until halt.
constexpr unsigned run(const State *tm, const Tape &tape, unsigned state) {
  return state == halt ? 0 :
         run(tm, move(update(tape, tm[state][tape.val].tape),
                      tm[state][tape.val].dir),
             tm[state][tape.val].next) + 1;
}

// 3-state busy beaver. S(bb3) = 21.
constexpr State bb3[] = {
  { { true, R, 1 }, { true, R, halt } },
  { { true, L, 1 }, { false, R, 2 } },
  { { true, L, 2 }, { true, L, 0 } }
};
static_assert(run(bb3, Tape(), 0) == 21, "");

// 4-state busy beaver. S(bb4) = 107.
constexpr State bb4[] = {
  { { true, R, 1 }, { true, L, 1 } },
  { { true, L, 0 }, { false, L, 2 } },
  { { true, R, halt }, { true, L, 3 } },
  { { true, R, 3 }, { false, R, 0 } } };
static_assert(run(bb4, Tape(), 0) == 107, "");
