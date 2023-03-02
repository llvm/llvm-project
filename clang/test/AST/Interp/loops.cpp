// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++14 -verify %s
// RUN: %clang_cc1 -std=c++14 -verify=ref %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify=expected-cpp20 %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s

// ref-no-diagnostics
// expected-no-diagnostics
// expected-cpp20-no-diagnostics

namespace WhileLoop {
  constexpr int f() {
    int i = 0;
    while(false) {
      i = i + 1;
    }
    return i;
  }
  static_assert(f() == 0, "");


  constexpr int f2() {
    int i = 0;
    while(i != 5) {
      i = i + 1;
    }
    return i;
  }
  static_assert(f2() == 5, "");

  constexpr int f3() {
    int i = 0;
    while(true) {
      i = i + 1;

      if (i == 5)
        break;
    }
    return i;
  }
  static_assert(f3() == 5, "");

  constexpr int f4() {
    int i = 0;
    while(i != 5) {

      i = i + 1;
      continue;
      i = i - 1;
    }
    return i;
  }
  static_assert(f4() == 5, "");


  constexpr int f5(bool b) {
    int i = 0;

    while(true) {
      if (!b) {
        if (i == 5)
          break;
      }

      if (b) {
        while (i != 10) {
          i = i + 1;
          if (i == 8)
            break;

          continue;
        }
      }

      if (b)
        break;

      i = i + 1;
      continue;
    }

    return i;
  }
  static_assert(f5(true) == 8, "");
  static_assert(f5(false) == 5, "");

#if 0
  /// FIXME: This is an infinite loop, which should
  ///   be rejected.
  constexpr int f6() {
    while(true);
  }
#endif
};

namespace DoWhileLoop {

  constexpr int f() {
    int i = 0;
    do {
      i = i + 1;
    } while(false);
    return i;
  }
  static_assert(f() == 1, "");

  constexpr int f2() {
    int i = 0;
    do {
      i = i + 1;
    } while(i != 5);
    return i;
  }
  static_assert(f2() == 5, "");


  constexpr int f3() {
    int i = 0;
    do {
      i = i + 1;
      if (i == 5)
        break;
    } while(true);
    return i;
  }
  static_assert(f3() == 5, "");

  constexpr int f4() {
    int i = 0;
    do {
      i = i + 1;
      continue;
      i = i - 1;
    } while(i != 5);
    return i;
  }
  static_assert(f4() == 5, "");

  constexpr int f5(bool b) {
    int i = 0;

    do {
      if (!b) {
        if (i == 5)
          break;
      }

      if (b) {
        do {
          i = i + 1;
          if (i == 8)
            break;

          continue;
        } while (i != 10);
      }

      if (b)
        break;

      i = i + 1;
      continue;
    } while(true);

    return i;
  }
  static_assert(f5(true) == 8, "");
  static_assert(f5(false) == 5, "");

#if __cplusplus >= 202002L
  constexpr int f6() {
    int i;
    do {
      i = 5;
      break;
    } while (true);
    return i;
  }
  static_assert(f6() == 5, "");
#endif

#if 0
  /// FIXME: This is an infinite loop, which should
  ///   be rejected.
  constexpr int f7() {
    while(true);
  }
#endif
};

namespace ForLoop {
  constexpr int f() {
    int i = 0;
    for (;false;) {
      i = i + 1;
    }
    return i;
  }
  static_assert(f() == 0, "");

  constexpr int f2() {
    int m = 0;
    for (int i = 0; i < 10; i = i + 1){
      m = i;
    }
    return m;
  }
  static_assert(f2() == 9, "");

  constexpr int f3() {
    int i = 0;
    for (; i != 5; i = i + 1);
    return i;
  }
  static_assert(f3() == 5, "");

  constexpr int f4() {
    int i = 0;
    for (;;) {
      i = i + 1;

      if (i == 5)
        break;
    }
    return i;
  }
  static_assert(f4() == 5, "");

  constexpr int f5() {
    int i = 0;
    for (;i != 5;) {
      i = i + 1;
      continue;
      i = i - 1;
    }
    return i;
  }
  static_assert(f5() == 5, "");

  constexpr int f6(bool b) {
    int i = 0;

    for (;true;) {
      if (!b) {
        if (i == 5)
          break;
      }

      if (b) {
        for (; i != 10; i = i + 1) {
          if (i == 8)
            break;
          continue;
        }
      }

      if (b)
        break;

      i = i + 1;
      continue;
    }

    return i;
  }
  static_assert(f6(true) == 8, "");
  static_assert(f6(false) == 5, "");

#if 0
  /// FIXME: This is an infinite loop, which should
  ///   be rejected.
  constexpr int f6() {
    for(;;);
  }
#endif

};
