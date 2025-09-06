// RUN: %check_clang_tidy %s bugprone-loop-variable-copied-then-modified %t -- -- -I%S -std=c++!4 -config="{CheckOptions: {bugprone-loop-variable-copied-then-modified.IgnoreInexpensiveVariables: true}}"

#include "Inputs/system-header-simulator/sim_set"
#include "Inputs/system-header-simulator/sim_unordered_set"
#include "Inputs/system-header-simulator/sim_map"
#include "Inputs/system-header-simulator/sim_unordered_map"
#include "Inputs/system-header-simulator/sim_vector"
#include "Inputs/system-header-simulator/sim_algorithm"

template <typename T>
struct Iterator {
  void operator++() {}
  const T& operator*() {
    static T* TT = new T();
    return *TT;
  }
  bool operator!=(const Iterator &) { return false; }
};
template <typename T>
struct View {
  T begin() { return T(); }
  T begin() const { return T(); }
  T end() { return T(); }
  T end() const { return T(); }
};

struct S {
  int value;

  S() : value(0) {};
  S(const S &);
  ~S();
  S &operator=(const S &);
  void modify() {
    value++;
  }
};

void NegativeOnlyCopyingInts() {
    std::vector<int> foo;
    foo.push_back(1);
    foo.push_back(2);
    foo.push_back(3);
    for (int v : foo) {
        v += 1;
    }
}

void PositiveLoopVariableCopiedAndThenModfied() {
  for (S S1 : View<Iterator<S>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: loop variable 'S1' is copied and then modified, which is likely a bug; you probably want to modify the underlying object and not this copy. If you *did* intend to modify this copy, please use an explicit copy inside the body of the loop
    // CHECK-FIXES: for (const S& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}