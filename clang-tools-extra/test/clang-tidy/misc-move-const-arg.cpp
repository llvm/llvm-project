// RUN: %check_clang_tidy %s misc-move-const-arg %t

namespace std {
template <typename> struct remove_reference;

template <typename _Tp> struct remove_reference { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &> { typedef _Tp type; };

template <typename _Tp> struct remove_reference<_Tp &&> { typedef _Tp type; };

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t);

} // namespace std

class A {
public:
  A() {}
  A(const A &rhs) {}
  A(A &&rhs) {}
};

int f1() {
  return std::move(42);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the expression of the trivially-copyable type 'int' has no effect; remove std::move() [misc-move-const-arg]
  // CHECK-FIXES: return 42;
}

int f2(int x2) {
  return std::move(x2);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 'x2' of the trivially-copyable type 'int'
  // CHECK-FIXES: return x2;
}

int *f3(int *x3) {
  return std::move(x3);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the variable 'x3' of the trivially-copyable type 'int *'
  // CHECK-FIXES: return x3;
}

A f4(A x4) { return std::move(x4); }

A f5(const A x5) {
  return std::move(x5);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the const variable 'x5' has no effect; remove std::move() or make the variable non-const [misc-move-const-arg]
  // CHECK-FIXES: return x5;
}

template <typename T> T f6(const T x6) { return std::move(x6); }

void f7() { int a = f6(10); }

#define M1(x) x
void f8() {
  const A a;
  M1(A b = std::move(a);)
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: std::move of the const variable 'a' has no effect; remove std::move() or make the variable non-const
  // CHECK-FIXES: M1(A b = a;)
}

#define M2(x) std::move(x)
int f9() { return M2(1); }

template <typename T> T f10(const int x10) {
  return std::move(x10);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: std::move of the const variable 'x10' of the trivially-copyable type 'const int' has no effect; remove std::move() [misc-move-const-arg]
  // CHECK-FIXES: return x10;
}
void f11() {
  f10<int>(1);
  f10<double>(1);
}

class NoMoveSemantics {
 public:
  NoMoveSemantics();
  NoMoveSemantics(const NoMoveSemantics &);

  NoMoveSemantics &operator=(const NoMoveSemantics &);
};

void callByConstRef(const NoMoveSemantics &);
void callByConstRef(int i, const NoMoveSemantics &);

void moveToConstReferencePositives() {
  NoMoveSemantics obj;

  // Basic case.
  callByConstRef(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing result of std::move() as
  // CHECK-FIXES: callByConstRef(obj);

  // Also works for second argument.
  callByConstRef(1, std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: passing result of std::move() as
  // CHECK-FIXES: callByConstRef(1, obj);

  // Works if std::move() applied to a temporary.
  callByConstRef(std::move(NoMoveSemantics()));
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: passing result of std::move() as
  // CHECK-FIXES: callByConstRef(NoMoveSemantics());

  // Works if calling a copy constructor.
  NoMoveSemantics other(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing result of std::move() as
  // CHECK-FIXES: NoMoveSemantics other(obj);

  // Works if calling assignment operator.
  other = std::move(obj);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing result of std::move() as
  // CHECK-FIXES: other = obj;
}

class MoveSemantics {
 public:
  MoveSemantics();
  MoveSemantics(MoveSemantics &&);

  MoveSemantics &operator=(MoveSemantics &&);
};

void callByValue(MoveSemantics);

void callByRValueRef(MoveSemantics &&);

template <class T>
void templateFunction(T obj) {
  T other = std::move(obj);
}

#define M3(T, obj)            \
  do {                        \
    T other = std::move(obj); \
  } while (true)

#define CALL(func) (func)()

void moveToConstReferenceNegatives() {
  // No warning when actual move takes place.
  MoveSemantics move_semantics;
  callByValue(std::move(move_semantics));
  callByRValueRef(std::move(move_semantics));
  MoveSemantics other(std::move(move_semantics));
  other = std::move(move_semantics);

  // No warning if std::move() not used.
  NoMoveSemantics no_move_semantics;
  callByConstRef(no_move_semantics);

  // No warning if instantiating a template.
  templateFunction(no_move_semantics);

  // No warning inside of macro expansions.
  M3(NoMoveSemantics, no_move_semantics);

  // No warning inside of macro expansion, even if the macro expansion is inside
  // a lambda that is, in turn, an argument to a macro.
  CALL([no_move_semantics] { M3(NoMoveSemantics, no_move_semantics); });
}

class MoveOnly {
public:
  MoveOnly(const MoveOnly &other) = delete;
  MoveOnly &operator=(const MoveOnly &other) = delete;
  MoveOnly(MoveOnly &&other) = default;
  MoveOnly &operator=(MoveOnly &&other) = default;
};
template <class T>
void Q(T);
void moveOnlyNegatives(MoveOnly val) {
  Q(std::move(val));
}
