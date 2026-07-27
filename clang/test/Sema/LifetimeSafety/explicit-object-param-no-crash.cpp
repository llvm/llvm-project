// RUN: %clang_cc1 %s -std=c++23 -verify -fsyntax-only -Wlifetime-safety -Wlifetime-safety-lifetimebound-violation

// expected-no-diagnostics

// Explicit object member functions must not be treated as having an implicit
// object argument.
struct Foo {
  template <typename T>
  int get(this Foo &&self, T) {
    return self.field;
  }

  int field;
};

struct Base {};
struct Derived : Base {
  int base(this Base &&, int);
};

template <typename T>
struct Crtp {
  template <typename Self>
  int deduced(this Self &&self, T) {
    return self.field;
  }
};

struct Widget : Crtp<int> {
  int field;
};

struct NoMove {
  NoMove() = default;
  NoMove(NoMove &&) = delete;
  NoMove(const NoMove &) = delete;

  int get(this NoMove &&self, int) {
    return self.field;
  }

  int field;
};

template <typename T>
struct SharedPtr {
  SharedPtr() = default;
  SharedPtr(SharedPtr &&o) : p(o.p) { o.p = nullptr; }
  SharedPtr(const SharedPtr &) = default;

  int get(this SharedPtr &&self, int) {
    return self.p->field;
  }

  T *p = nullptr;
};

void call() {
  Foo().get(0);
  Derived().base(0);
  Widget().deduced(0);
  NoMove().get(0);
  SharedPtr<NoMove>().get(0);
}

struct Holder {
  int field;
  const int *borrow() [[clang::lifetimebound]] { return &field; }
  void consume(this Holder &&self) {}
};

const int *object_arg_is_not_moved(Holder &&h [[clang::lifetimebound]]) {
  static_cast<Holder &&>(h).consume();
  return h.borrow();
}
