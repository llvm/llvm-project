// RUN: %check_clang_tidy %s bugprone-loop-variable-copied-then-modified %t --fix-notes \
// RUN: -config="{CheckOptions: \
// RUN: {bugprone-loop-variable-copied-then-modified.WarnOnlyOnAutoCopies: true}}" \
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

void NegativeLoopVariableIsNotAuto() {
  for (S S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}

void PositiveLoopVariableIsAuto() {
  for (auto S1 : View<Iterator<S>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable 'S1' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-MESSAGES: [[@LINE-2]]:13: note: consider making 'S1' a reference
    // CHECK-FIXES: for (const auto& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}
