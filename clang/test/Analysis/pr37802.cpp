// RUN: %clang_analyze_cc1 -w -analyzer-checker=core -verify %s

// expected-no-diagnostics

typedef __typeof(sizeof(int)) size_t;
void *operator new(size_t, void *h) { return h; }

// I've no idea what this code does, but it used to crash, so let's keep it.
namespace pr37802_v1 {
struct J {
  int *p;
};
class X {
  void *ar;

public:
  X(void *t) : ar(t) {}
  template <typename T>
  void f(const T &t) {
    new (ar) T(t);
  }
};
class Y {
public:
  template <typename T>
  void f(T &&);
  void f(J t) {
    f(*t.p);
  }
};
class Z {
  int at() const {}

public:
  Z(const Z &other) {
    other.au(X(this));
  }
  template <typename T>
  void au(T t) const {
    void *c = const_cast<Z *>(this);
    if (at()) {
      t.f(*static_cast<J *>(c));
    } else {
      t.f(*static_cast<bool *>(c));
    }
  }
};
Z g() {
  Z az = g();
  Z e = az;
  Y d;
  e.au(d);
}
} // namespace pr37802_v1


// This slightly modified code crashed differently.
namespace pr37802_v2 {
struct J {
  int *p;
};

class X {
  void *ar;

public:
  X(void *t) : ar(t) {}
  void f(const J &t) { new (ar) J(t); }
  void f(const bool &t) { new (ar) bool(t); }
};

class Y {
public:
  void boolf(bool &&);
  void f(J &&);
  void f(J t) { boolf(*t.p); }
};

class Z {
  int at() const {}

public:
  Z(const Z &other) { other.au(X(this)); }
  void au(X t) const {
    void *c = const_cast<Z *>(this);
    if (at()) {
      t.f(*static_cast<J *>(c));
    } else {
      t.f(*static_cast<bool *>(c));
    }
  }
  void au(Y t) const {
    void *c = const_cast<Z *>(this);
    if (at()) {
      t.f(*static_cast<J *>(c));
    } else {
    }
  }
};

Z g() {
  Z az = g();
  Z e = az;
  Y d;
  e.au(d);
}
} // namespace pr37802_v2
