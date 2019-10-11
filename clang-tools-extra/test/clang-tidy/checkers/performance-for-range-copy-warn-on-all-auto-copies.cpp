// RUN: %check_clang_tidy %s performance-for-range-copy %t -- \
// RUN:     -config="{CheckOptions: [{key: "performance-for-range-copy.WarnOnAllAutoCopies", value: 1}]}"

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
  S();
  S(const S &);
  ~S();
  S &operator=(const S &);
};

void NegativeLoopVariableNotAuto() {
  for (S S1 : View<Iterator<S>>()) {
    S* S2 = &S1;
  }
}

void PositiveTriggeredForAutoLoopVariable() {
  for (auto S1 : View<Iterator<S>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: the loop variable's type is not a reference type; this creates a copy in each iteration; consider making this a reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& S1 : View<Iterator<S>>()) {
    S* S2 = &S1;
  }
}
