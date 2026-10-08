// RUN: %check_clang_tidy %s performance-move-const-arg %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  {performance-move-const-arg.CheckTriviallyCopyableMove: false}}'

#include <utility>

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

  // Basic case. It is here just to have a single "detected and fixed" case.
  callByConstRef(std::move(obj));
  // CHECK-MESSAGES: :[[@LINE-1]]:18:  warning: passing result of std::move() as a const reference argument; no move will actually happen [performance-move-const-arg]
  // CHECK-FIXES: callByConstRef(obj);
}

struct TriviallyCopyable {
  int i;
};

void f(TriviallyCopyable) {}

void g() {
  TriviallyCopyable obj;
  f(std::move(obj));
}

class MoveSemantics {
 public:
  MoveSemantics();
  MoveSemantics(MoveSemantics &&);

  MoveSemantics &operator=(MoveSemantics &&);
};

void fmovable(MoveSemantics);

void lambda1() {
  auto f = [](MoveSemantics m) {
    fmovable(std::move(m));
  };
  f(MoveSemantics());
}

template<class T> struct function {};

template<typename Result, typename... Args>
class function<Result(Args...)> {
public:
  function() = default;
  void operator()(Args... args) const {
    fmovable(std::forward<Args>(args)...);
  }
};

void functionInvocation() {
  function<void(MoveSemantics)> callback;
  MoveSemantics m;
  callback(std::move(m));
}

void lambda2() {
  function<void(MoveSemantics)> callback;

  auto f = [callback = std::move(callback)](MoveSemantics m) mutable {
    callback(std::move(m));
  };
  f(MoveSemantics());
}
