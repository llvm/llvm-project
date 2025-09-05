// RUN: %check_clang_tidy %s bugprone-loop-variable-copied-then-modified %t --fix-notes \
// RUN: -config="{CheckOptions: \
// RUN: {bugprone-loop-variable-copied-then-modified.IgnoreInexpensiveVariables: true}}" \
// RUN: -- -I%S
#include "Inputs/system-header-simulator/sim_initializer_list"
#include "Inputs/system-header-simulator/sim_vector"

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
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: loop variable 'S1' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-MESSAGES: [[@LINE-2]]:10: note: consider making 'S1' a reference
    // CHECK-FIXES: for (const S& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}
