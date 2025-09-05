// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-loop-variable-copied-then-modified %t

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

template <typename V>
struct Generic {
  V value;

  Generic() : value{} {};
  Generic(const Generic &);
  ~Generic();
  Generic &operator=(const Generic &);
  void modify() {
    value++;
  }
};

void NegativeLoopVariableNotCopied() {
  for (const S& S1 : View<Iterator<S>>()) {
    // It's fine to copy-by-value S1 into some other S.
    S S2 = S1;
  }
}

void NegativeLoopVariableCopiedButNotModified() {
  for (S S1 : View<Iterator<S>>()) {
  }
}

void PositiveLoopVariableCopiedAndThenModfied() {
  for (S S1 : View<Iterator<S>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: loop variable 'S1' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-FIXES: for (const S& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}

void PositiveLoopVariableCopiedAndThenModifiedAuto() {
  for (auto S1 : View<Iterator<S>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable 'S1' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-FIXES: for (const auto& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}

void PositiveLoopVariableCopiedAndThenModfiedGeneric() {
  for (Generic G : View<Iterator<Generic<double>>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:16: warning: loop variable 'G' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-FIXES: for (const Generic& G : View<Iterator<Generic<double>>>()) {
    G.modify();
  }
}
