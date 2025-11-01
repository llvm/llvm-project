// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-loop-variable-copied-then-modified %t --fix-notes

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

void PositiveLoopVariableCopiedAndThenModfiedGeneric() {
  for (Generic G : View<Iterator<Generic<double>>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:16: warning: loop variable 'G' is copied and then (possibly) modified; use an explicit copy inside the body of the loop or make the variable a reference
    // CHECK-MESSAGES: [[@LINE-2]]:16: note: consider making 'G' a reference
    // CHECK-FIXES: for (const Generic& G : View<Iterator<Generic<double>>>()) {
    G.modify();
  }
}

void NegativeLoopVariableIsReferenceAndModifiedGeneric() {
  for (const Generic<double>& G : View<Iterator<Generic<double>>>()) {
    // It's fine to copy-by-value G into some other G.
    Generic<double> G2 = G;
  }
}
