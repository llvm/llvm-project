// RUN: %check_clang_tidy %s bugprone-loop-variable-copied-then-modified %t

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
    // CHECK-MESSAGES: [[@LINE-1]]:10: warning: loop variable 'S1' is copied and then modified, which is likely a bug; you probably want to modify the underlying object and not this copy. If you *did* intend to modify this copy, please use an explicit copy inside the body of the loop
    // CHECK-FIXES: for (const S& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}
