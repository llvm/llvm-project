// RUN: %clang_cc1 -std=c++1z -verify %s

namespace pr32864_0 {
  struct transfer_t {
    void *fctx;
  };
  template <typename Ctx> class record {
    void run() {
      transfer_t t;
      Ctx from{t.fctx};
    }
  };
}

namespace pr33140_0a {
  struct S {
    constexpr S(const int &a = 0) {}
  };
  void foo(void) { S s[2] = {}; }
}

namespace pr33140_0b {
  bool bar(float const &f = 0);
  bool foo() { return bar() && bar(); }
}

namespace pr33140_2 {
  struct A { int &&r = 0; };
  struct B { A x, y; };
  B b = {};
}

namespace pr33140_3 {
  typedef struct Y { unsigned int c; } Y_t;
  struct X {
    Y_t a;
  };
  struct X foo[2] = {[0 ... 1] = {.a = (Y_t){.c = 0}}}; // expected-warning {{C99 extension}}
}

namespace pr33140_6 {
  struct Y { unsigned int c; };
  struct X { struct Y *p; };
  int f() {
    // FIXME: This causes clang to crash.
    //return (struct X[2]){ [0 ... 1] = { .p = &(struct Y&)(struct Y&&)(struct Y){0} } }[0].p->c;
    return 0;
  }
}

namespace pr33140_10 {
  int a(const int &n = 0);
  bool b() { return a() == a(); }
}

namespace GH67317 {
struct array {
  int (&data)[2];
  array() : data(*new int[1][2]) {}
};
}

namespace GH96670 {
inline constexpr long ullNil = -1;

template<typename T = long, const T &Nil = ullNil>
struct Test {};

inline constexpr long lNil = -1;
Test<long, lNil> c;
}
