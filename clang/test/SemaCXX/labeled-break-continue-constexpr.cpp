// RUN: %clang_cc1 -fnamed-loops -std=c++23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fnamed-loops -std=c++23 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

struct Tracker {
  bool& destroyed;
  constexpr Tracker(bool& destroyed) : destroyed{destroyed} {}
  constexpr ~Tracker() { destroyed = true; }
};

constexpr int f1() {
  a: for (;;) {
    for (;;) {
      break a;
    }
  }
  return 1;
}
static_assert(f1() == 1);

constexpr int f2() {
  int x{};
  a: for (int i = 0; i < 10; i++) {
    b: for (int j = 0; j < 10; j++) {
      x += j;
      if (i == 2 && j == 2) break a;
    }
  }
  return x;
}
static_assert(f2() == 93);

constexpr int f3() {
  int x{};
  a: for (int i = 0; i < 10; i++) {
    x += i;
    continue a;
  }
  return x;
}
static_assert(f3() == 45);

constexpr int f4() {
  int x{};
  a: for (int i = 1; i < 10; i++) {
    x += i;
    break a;
  }
  return x;
}
static_assert(f4() == 1);

constexpr bool f5(bool should_break) {
  bool destroyed = false;
  a: while (!destroyed) {
    while (true) {
      Tracker _{destroyed};
      if (should_break) break a;
      continue a;
    }
  }
  return destroyed;
}
static_assert(f5(true));
static_assert(f5(false));

constexpr bool f6(bool should_break) {
  bool destroyed = false;
  a: while (!destroyed) {
    while (true) {
      while (true) {
        Tracker _{destroyed};
        while (true) {
          while (true) {
            if (should_break) break a;
            continue a;
          }
        }
      }
    }
  }
  return destroyed;
}
static_assert(f6(true));
static_assert(f6(false));

constexpr int f7(bool should_break) {
  int x = 100;
  a: for (int i = 0; i < 10; i++) {
    b: switch (1) {
      case 1:
        x += i;
        if (should_break) break a;
        break b;
    }
  }
  return x;
}
static_assert(f7(true) == 100);
static_assert(f7(false) == 145);

constexpr bool f8() {
  a: switch (1) {
    case 1: {
      while (true) {
        switch (1) {
          case 1: break a;
        }
      }
    }
  }
  return true;
}
static_assert(f8());

constexpr bool f9() {
  a: do {
    while (true) {
      break a;
    }
  } while (true);
  return true;
}
static_assert(f9());

constexpr int f10(bool should_break) {
  int a[10]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  int x{};
  a: for (int v : a) {
    for (int i = 0; i < 3; i++) {
      x += v;
      if (should_break && v == 5) break a;
    }
  }
  return x;
}

static_assert(f10(true) == 35);
static_assert(f10(false) == 165);

constexpr bool f11() {
  struct X {
    int a[10]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    Tracker t;
    constexpr X(bool& b) : t{b} {}
  };

  bool destroyed = false;
  a: for (int v : X(destroyed).a) {
    for (int i = 0; i < 3; i++) {
      if (v == 5) break a;
    }
  }
  return destroyed;
}
static_assert(f11());

template <typename T>
constexpr T f12() {
  T x{};
  a: for (T i = 0; i < 10; i++) {
    b: for (T j = 0; j < 10; j++) {
      x += j;
      if (i == 2 && j == 2) break a;
    }
  }
  return x;
}
static_assert(f12<int>() == 93);
static_assert(f12<unsigned>() == 93u);
