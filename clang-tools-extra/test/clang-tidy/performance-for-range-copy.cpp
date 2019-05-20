// RUN: %check_clang_tidy %s performance-for-range-copy %t -- -- -fno-delayed-template-parsing

namespace std {

template <typename _Tp>
struct remove_reference { typedef _Tp type; };
template <typename _Tp>
struct remove_reference<_Tp&> { typedef _Tp type; };
template <typename _Tp>
struct remove_reference<_Tp&&> { typedef _Tp type; };

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) {
  return static_cast<typename std::remove_reference<_Tp>::type &&>(__t);
}

} // std

template <typename T>
struct Iterator {
  void operator++() {}
  const T& operator*() {
    static T* TT = new T();
    return *TT;
  }
  bool operator!=(const Iterator &) { return false; }
  typedef const T& const_reference;
};
template <typename T>
struct View {
  T begin() { return T(); }
  T begin() const { return T(); }
  T end() { return T(); }
  T end() const { return T(); }
  typedef typename T::const_reference const_reference;
};

struct ConstructorConvertible {
};

struct S {
  S();
  S(const S &);
  S(const ConstructorConvertible&) {}
  ~S();
  S &operator=(const S &);
};

struct Convertible {
  operator S() const {
    return S();
  }
};

void negativeConstReference() {
  for (const S &S1 : View<Iterator<S>>()) {
  }
}

void negativeUserDefinedConversion() {
  Convertible C[0];
  for (const S &S1 : C) {
  }
}

void negativeImplicitConstructorConversion() {
  ConstructorConvertible C[0];
  for (const S &S1 : C) {
  }
}

template <typename T>
void uninstantiated() {
  for (const S S1 : View<Iterator<S>>()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: the loop variable's type is not a reference type; this creates a copy in each iteration; consider making this a reference [performance-for-range-copy]
  // CHECK-FIXES: {{^}}  for (const S& S1 : View<Iterator<S>>()) {}

  // Don't warn on dependent types.
  for (const T t1 : View<Iterator<T>>()) {
  }
}

template <typename T>
void instantiated() {
  for (const S S2 : View<Iterator<S>>()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: the loop variable's type is {{.*}}
  // CHECK-FIXES: {{^}}  for (const S& S2 : View<Iterator<S>>()) {}

  for (const T T2 : View<Iterator<T>>()) {}
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: the loop variable's type is {{.*}}
  // CHECK-FIXES: {{^}}  for (const T& T2 : View<Iterator<T>>()) {}
}

template <typename T>
void instantiatedNegativeTypedefConstReference() {
  for (typename T::const_reference T2 : T()) {
    S S1 = T2;
  }
}

void f() {
  instantiated<int>();
  instantiated<S>();
  instantiatedNegativeTypedefConstReference<View<Iterator<S>>>();
}

struct Mutable {
  Mutable() {}
  Mutable(const Mutable &) = default;
  Mutable(Mutable&&) = default;
  Mutable(const Mutable &, const Mutable &) {}
  void setBool(bool B) {}
  bool constMethod() const {
    return true;
  }
  Mutable& operator[](int I) {
    return *this;
  }
  bool operator==(const Mutable &Other) const {
    return true;
  }
  ~Mutable() {}
};

struct Point {
  ~Point() {}
  int x, y;
};

Mutable& operator<<(Mutable &Out, bool B) {
  Out.setBool(B);
  return Out;
}

bool operator!=(const Mutable& M1, const Mutable& M2) {
  return false;
}

void use(const Mutable &M);
void use(int I);
void useTwice(const Mutable &M1, const Mutable &M2);
void useByValue(Mutable M);
void useByConstValue(const Mutable M);
void mutate(Mutable *M);
void mutate(Mutable &M);
void onceConstOnceMutated(const Mutable &M1, Mutable &M2);

void negativeVariableIsMutated() {
  for (auto M : View<Iterator<Mutable>>()) {
    mutate(M);
  }
  for (auto M : View<Iterator<Mutable>>()) {
    mutate(&M);
  }
  for (auto M : View<Iterator<Mutable>>()) {
    M.setBool(true);
  }
}

void negativeOnceConstOnceMutated() {
  for (auto M : View<Iterator<Mutable>>()) {
    onceConstOnceMutated(M, M);
  }
}

void negativeVarIsMoved() {
  for (auto M : View<Iterator<Mutable>>()) {
    auto Moved = std::move(M);
  }
}

void negativeNonConstOperatorIsInvoked() {
  for (auto NonConstOperatorInvokee : View<Iterator<Mutable>>()) {
    auto& N = NonConstOperatorInvokee[0];
  }
}

void negativeNonConstNonMemberOperatorInvoked() {
  for (auto NonConstOperatorInvokee : View<Iterator<Mutable>>()) {
    NonConstOperatorInvokee << true;
  }
}

void negativeConstCheapToCopy() {
  for (const int I : View<Iterator<int>>()) {
  }
}

void negativeConstCheapToCopyTypedef() {
  typedef const int ConstInt;
  for (ConstInt C  : View<Iterator<ConstInt>>()) {
  }
}

void negativeCheapToCopy() {
  for (int I : View<Iterator<int>>()) {
    use(I);
  }
}

void negativeCheapToCopyTypedef() {
  typedef int Int;
  for (Int I : View<Iterator<Int>>()) {
    use(I);
  }
}

void positiveOnlyConstMethodInvoked() {
  for (auto M : View<Iterator<Mutable>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& M : View<Iterator<Mutable>>()) {
    M.constMethod();
  }
}

void positiveOnlyUsedAsConstArguments() {
  for (auto UsedAsConst : View<Iterator<Mutable>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& UsedAsConst : View<Iterator<Mutable>>()) {
    use(UsedAsConst);
    useTwice(UsedAsConst, UsedAsConst);
    useByValue(UsedAsConst);
    useByConstValue(UsedAsConst);
  }
}

void positiveOnlyAccessedFieldAsConst() {
  for (auto UsedAsConst : View<Iterator<Point>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& UsedAsConst : View<Iterator<Point>>()) {
    use(UsedAsConst.x);
    use(UsedAsConst.y);
  }
}

void positiveOnlyUsedInCopyConstructor() {
  for (auto A : View<Iterator<Mutable>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& A : View<Iterator<Mutable>>()) {
    Mutable Copy = A;
    Mutable Copy2(A);
  }
}

void positiveTwoConstConstructorArgs() {
  for (auto A : View<Iterator<Mutable>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& A : View<Iterator<Mutable>>()) {
    Mutable Copy(A, A);
  }
}

void PositiveConstMemberOperatorInvoked() {
  for (auto ConstOperatorInvokee : View<Iterator<Mutable>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& ConstOperatorInvokee : View<Iterator<Mutable>>()) {
    bool result = ConstOperatorInvokee == Mutable();
  }
}

void PositiveConstNonMemberOperatorInvoked() {
  for (auto ConstOperatorInvokee : View<Iterator<Mutable>>()) {
    // CHECK-MESSAGES: [[@LINE-1]]:13: warning: loop variable is copied but only used as const reference; consider making it a const reference [performance-for-range-copy]
    // CHECK-FIXES: for (const auto& ConstOperatorInvokee : View<Iterator<Mutable>>()) {
    bool result = ConstOperatorInvokee != Mutable();
  }
}

void IgnoreLoopVariableNotUsedInLoopBody() {
  for (auto _ : View<Iterator<S>>()) {
  }
}
