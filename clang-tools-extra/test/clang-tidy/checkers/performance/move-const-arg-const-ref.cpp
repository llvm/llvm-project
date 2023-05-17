// RUN: %check_clang_tidy %s performance-move-const-arg %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: performance-move-const-arg.CheckMoveToConstRef, value: false}]}'

namespace std {
template <typename>
struct remove_reference;

template <typename _Tp>
struct remove_reference {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &> {
  typedef _Tp type;
};

template <typename _Tp>
struct remove_reference<_Tp &&> {
  typedef _Tp type;
};

template <typename _Tp>
constexpr typename std::remove_reference<_Tp>::type &&move(_Tp &&__t) {
  return static_cast<typename std::remove_reference<_Tp>::type &&>(__t);
}

template <typename _Tp>
constexpr _Tp &&
forward(typename remove_reference<_Tp>::type &__t) noexcept {
  return static_cast<_Tp &&>(__t);
}

} // namespace std

struct TriviallyCopyable {
  int i;
};

void f(TriviallyCopyable) {}

void g() {
  TriviallyCopyable obj;
  // Some basic test to ensure that other warnings from
  // performance-move-const-arg are still working and enabled.
  f(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: std::move of the variable 'obj' of the trivially-copyable type 'TriviallyCopyable' has no effect; remove std::move() [performance-move-const-arg]
  // CHECK-FIXES: f(obj);
}

class NoMoveSemantics {
public:
  NoMoveSemantics();
  NoMoveSemantics(const NoMoveSemantics &);
  NoMoveSemantics &operator=(const NoMoveSemantics &);
};

class MoveSemantics {
public:
  MoveSemantics();
  MoveSemantics(MoveSemantics &&);

  MoveSemantics &operator=(MoveSemantics &&);
};

void callByConstRef1(const NoMoveSemantics &);
void callByConstRef2(const MoveSemantics &);

void moveToConstReferencePositives() {
  NoMoveSemantics a;

  // This call is now allowed since CheckMoveToConstRef is false.
  callByConstRef1(std::move(a));

  MoveSemantics b;

  // This call is now allowed since CheckMoveToConstRef is false.
  callByConstRef2(std::move(b));
}
