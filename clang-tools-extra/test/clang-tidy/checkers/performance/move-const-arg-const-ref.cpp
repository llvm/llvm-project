// RUN: %check_clang_tidy %s performance-move-const-arg %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  {performance-move-const-arg.CheckMoveToConstRef: false}}'

#include <utility>

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
