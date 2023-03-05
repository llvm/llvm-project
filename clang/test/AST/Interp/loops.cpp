// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++14 -verify %s
// RUN: %clang_cc1 -std=c++14 -verify=ref %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify=expected-cpp20 %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s

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

namespace RangeForLoop {
  constexpr int localArray() {
    int a[] = {1,2,3,4};
    int s = 0;
    for(int i : a) {
      s += i;
    }
    return s;
  }
  static_assert(localArray() == 10, "");

  constexpr int localArray2() {
    int a[] = {1,2,3,4};
    int s = 0;
    for(const int &i : a) {
      s += i;
    }
    return s;
  }
  static_assert(localArray2() == 10, "");

  constexpr int nested() {
    int s = 0;
    for (const int i : (int[]){1,2,3,4}) {
      int a[] = {i, i};
      for(int m : a) {
        s += m;
      }
    }
    return s;
  }
  static_assert(nested() == 20, "");

  constexpr int withBreak() {
    int s = 0;
    for (const int &i: (bool[]){false, true}) {
      if (i)
        break;
      s++;
    }
    return s;
  }
  static_assert(withBreak() == 1, "");

  constexpr void NoBody() {
    for (const int &i: (bool[]){false, true}); // expected-warning {{empty body}} \
                                               // expected-note {{semicolon on a separate line}} \
                                               // expected-cpp20-warning {{empty body}} \
                                               // expected-cpp20-note {{semicolon on a separate line}} \
                                               // ref-warning {{empty body}} \
                                               // ref-note {{semicolon on a separate line}}
  }
}
