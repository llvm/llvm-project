// RUN: %check_clang_tidy %s bugprone-loop-variable-copied-then-modified %t --fix-notes

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

struct ConstructorConvertible {
};

struct S {
  int value;

  S() : value(0) {};
  S(const S &);
  S(const ConstructorConvertible&) {}
  ~S();
  S &operator=(const S &);
  void modify() {
    value++;
  }
};

struct Convertible {
  operator S() const {
    return S();
  }
};

struct PairLike {
  int id;
  S data;
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

#define LOOP_AND_MODIFY(VAR_NAME, TYPE) \
  for (TYPE VAR_NAME : View<Iterator<TYPE>>()) { \
    VAR_NAME.modify(); \
  }

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
    // CHECK-MESSAGES: [[@LINE-2]]:10: note: consider making 'S1' a reference
    // CHECK-FIXES: for (const S& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}

void PositiveLoopVariableCopiedAndThenModifiedAuto() {
  for (auto S1 : View<Iterator<S>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable 'S1' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-MESSAGES: [[@LINE-2]]:13: note: consider making 'S1' a reference
    // CHECK-FIXES: for (const auto& S1 : View<Iterator<S>>()) {
    S1.modify();
  }
}

void PositiveLoopVariableIsTypedefType() {
  typedef int TypedefInt;
  for (TypedefInt ti : View<Iterator<TypedefInt>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:19: warning: loop variable 'ti' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-MESSAGES: [[@LINE-2]]:19: note: consider making 'ti' a reference
    // CHECK-FIXES: for (const TypedefInt& ti : View<Iterator<TypedefInt>>()) {
    ti += 1;
  }
}

void NegativeLoopVariableIsInsideMacro() {
  LOOP_AND_MODIFY(S1, S);
}

template <typename T>
struct ValueReturningIterator {
  void operator++() {}
  T operator*() { return T(); }
  bool operator!=(const ValueReturningIterator &) { return false; }
  typedef const T &const_reference;
};

void NegativeValueIterator() {
  // Check does not trigger for iterators that return elements by value.
  for (S S1 : View<ValueReturningIterator<S>>()) {
    S1.modify();
  }
}

void NegativeConstructedByConversion() {
  Convertible C[0];
  for (S S1 : C) {
    S1.modify();
  }
}

void NegativeNotConstructedByCopy() {
  // Designed to exercise the unless(NotConstructedByCopy) clause in the check's source code.
  // Distinct from NegativeConstructedByConversion because it exercises a converting constructor rather than a conversion operator.
  ConstructorConvertible C[0];
  for (S S1 : C) {
    S1.modify();
  }
}

void PositiveLoopVariableIsStructuredBinding() {
  for (auto [id, data] : View<Iterator<PairLike>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable '' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-MESSAGES: [[@LINE-2]]:13: note: consider making '' a reference
    // CHECK-FIXES: for (const auto& [id, data] : View<Iterator<PairLike>>()) {
    data.modify();
  }
}

void NegativeLoopVariableIsStructuredBinding() {
  for (const auto& [id, data] : View<Iterator<PairLike>>()) {
  }
}
