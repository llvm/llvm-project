// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++20 -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -std=c++17 -verify=cxx17 %s
// RUN: %clang_cc1 -std=c++20 -verify=ref %s
// RUN: %clang_cc1 -std=c++17 -verify=ref-cxx17 %s

#define INT_MIN (~__INT_MAX__)


namespace shifts {
  constexpr void test() { // ref-error {{constexpr function never produces a constant expression}} \
                          // ref-cxx17-error {{constexpr function never produces a constant expression}} \
                          // expected-error {{constexpr function never produces a constant expression}} \
                          // cxx17-error {{constexpr function never produces a constant expression}} \

    char c; // cxx17-warning {{uninitialized variable}} \
            // ref-cxx17-warning {{uninitialized variable}}

    c = 0 << 0;
    c = 0 << 1;
    c = 1 << 0;
    c = 1 << -0;
    c = 1 >> -0;
    c = 1 << -1; // expected-warning {{shift count is negative}} \
                 // expected-note {{negative shift count -1}} \
                 // cxx17-note {{negative shift count -1}} \
                 // cxx17-warning {{shift count is negative}} \
                 // ref-warning {{shift count is negative}} \
                 // ref-note {{negative shift count -1}} \
                 // ref-cxx17-warning {{shift count is negative}} \
                 // ref-cxx17-note {{negative shift count -1}}

    c = 1 >> -1; // expected-warning {{shift count is negative}} \
                 // cxx17-warning {{shift count is negative}} \
                 // ref-warning {{shift count is negative}} \
                 // ref-cxx17-warning {{shift count is negative}}
    c = 1 << (unsigned)-1; // expected-warning {{shift count >= width of type}} \
                           // FIXME: 'implicit conversion' warning missing in the new interpreter. \
                           // cxx17-warning {{shift count >= width of type}} \
                           // ref-warning {{shift count >= width of type}} \
                           // ref-warning {{implicit conversion}} \
                           // ref-cxx17-warning {{shift count >= width of type}} \
                           // ref-cxx17-warning {{implicit conversion}}
    c = 1 >> (unsigned)-1; // expected-warning {{shift count >= width of type}} \
                           // cxx17-warning {{shift count >= width of type}} \
                           // ref-warning {{shift count >= width of type}} \
                           // ref-cxx17-warning {{shift count >= width of type}}
    c = 1 << c;
    c <<= 0;
    c >>= 0;
    c <<= 1;
    c >>= 1;
    c <<= -1; // expected-warning {{shift count is negative}} \
              // cxx17-warning {{shift count is negative}} \
              // ref-warning {{shift count is negative}} \
              // ref-cxx17-warning {{shift count is negative}}
    c >>= -1; // expected-warning {{shift count is negative}} \
              // cxx17-warning {{shift count is negative}} \
              // ref-warning {{shift count is negative}} \
              // ref-cxx17-warning {{shift count is negative}}
    c <<= 999999; // expected-warning {{shift count >= width of type}} \
                  // cxx17-warning {{shift count >= width of type}} \
                  // ref-warning {{shift count >= width of type}} \
                  // ref-cxx17-warning {{shift count >= width of type}}
    c >>= 999999; // expected-warning {{shift count >= width of type}} \
                  // cxx17-warning {{shift count >= width of type}} \
                  // ref-warning {{shift count >= width of type}} \
                  // ref-cxx17-warning {{shift count >= width of type}}
    c <<= __CHAR_BIT__; // expected-warning {{shift count >= width of type}} \
                        // cxx17-warning {{shift count >= width of type}} \
                        // ref-warning {{shift count >= width of type}} \
                        // ref-cxx17-warning {{shift count >= width of type}}
    c >>= __CHAR_BIT__; // expected-warning {{shift count >= width of type}} \
                        // cxx17-warning {{shift count >= width of type}} \
                        // ref-warning {{shift count >= width of type}} \
                        // ref-cxx17-warning {{shift count >= width of type}}
    c <<= __CHAR_BIT__+1; // expected-warning {{shift count >= width of type}} \
                          // cxx17-warning {{shift count >= width of type}} \
                          // ref-warning {{shift count >= width of type}} \
                          // ref-cxx17-warning {{shift count >= width of type}}
    c >>= __CHAR_BIT__+1; // expected-warning {{shift count >= width of type}} \
                          // cxx17-warning {{shift count >= width of type}} \
                          // ref-warning {{shift count >= width of type}} \
                          // ref-cxx17-warning {{shift count >= width of type}}
    (void)((long)c << __CHAR_BIT__);

    int i; // cxx17-warning {{uninitialized variable}} \
           // ref-cxx17-warning {{uninitialized variable}}
    i = 1 << (__INT_WIDTH__ - 2);
    i = 2 << (__INT_WIDTH__ - 1); // cxx17-warning {{bits to represent, but 'int' only has}} \
                                  // ref-cxx17-warning {{bits to represent, but 'int' only has}}
    i = 1 << (__INT_WIDTH__ - 1); // cxx17-warning-not {{sets the sign bit of the shift expression}}
    i = -1 << (__INT_WIDTH__ - 1); // cxx17-warning {{shifting a negative signed value is undefined}} \
                                   // ref-cxx17-warning {{shifting a negative signed value is undefined}}
    i = -1 << 0; // cxx17-warning {{shifting a negative signed value is undefined}} \
                 // ref-cxx17-warning {{shifting a negative signed value is undefined}}
    i = 0 << (__INT_WIDTH__ - 1);
    i = (char)1 << (__INT_WIDTH__ - 2);

    unsigned u; // cxx17-warning {{uninitialized variable}} \
                // ref-cxx17-warning {{uninitialized variable}}
    u = 1U << (__INT_WIDTH__ - 1);
    u = 5U << (__INT_WIDTH__ - 1);

    long long int lli; // cxx17-warning {{uninitialized variable}} \
                       // ref-cxx17-warning {{uninitialized variable}}
    lli = INT_MIN << 2; // cxx17-warning {{shifting a negative signed value is undefined}} \
                        // ref-cxx17-warning {{shifting a negative signed value is undefined}}
    lli = 1LL << (sizeof(long long) * __CHAR_BIT__ - 2);
  }

  static_assert(1 << 4 == 16, "");
  constexpr unsigned m = 2 >> 1;
  static_assert(m == 1, "");
  constexpr unsigned char c = 0 << 8;
  static_assert(c == 0, "");
  static_assert(true << 1, "");
  static_assert(1 << (__INT_WIDTH__ +1) == 0, "");  // expected-error {{not an integral constant expression}} \
                                                    // expected-note {{>= width of type 'int'}} \
                                                    // cxx17-error {{not an integral constant expression}} \
                                                    // cxx17-note {{>= width of type 'int'}} \
                                                    // ref-error {{not an integral constant expression}} \
                                                    // ref-note {{>= width of type 'int'}} \
                                                    // ref-cxx17-error {{not an integral constant expression}} \
                                                    // ref-cxx17-note {{>= width of type 'int'}}

  constexpr int i1 = 1 << -1; // expected-error {{must be initialized by a constant expression}} \
                              // expected-note {{negative shift count -1}} \
                              // cxx17-error {{must be initialized by a constant expression}} \
                              // cxx17-note {{negative shift count -1}} \
                              // ref-error {{must be initialized by a constant expression}} \
                              // ref-note {{negative shift count -1}} \
                              // ref-cxx17-error {{must be initialized by a constant expression}} \
                              // ref-cxx17-note {{negative shift count -1}}

  constexpr int i2 = 1 << (__INT_WIDTH__ + 1); // expected-error {{must be initialized by a constant expression}} \
                                               // expected-note {{>= width of type}} \
                                               // cxx17-error {{must be initialized by a constant expression}} \
                                               // cxx17-note {{>= width of type}} \
                                               // ref-error {{must be initialized by a constant expression}} \
                                               // ref-note {{>= width of type}} \
                                               // ref-cxx17-error {{must be initialized by a constant expression}} \
                                               // ref-cxx17-note {{>= width of type}}

  constexpr char c2 = 1;
  constexpr int i3 = c2 << (__CHAR_BIT__ + 1); // Not ill-formed

  /// The purpose of these few lines is to test that we can shift more bits
  /// than an unsigned *of the host* has. There was a bug where we casted
  /// to host-unsigned. However, we cannot query what a host-unsigned even is
  /// here, so only test this on platforms where `sizeof(long long) > sizeof(unsigned)`.
  constexpr long long int L = 1;
  constexpr signed int R = (sizeof(unsigned) * 8) + 1;
  constexpr decltype(L) M  = (R > 32 && R < 64) ?  L << R : 0;
  constexpr decltype(L) M2 = (R > 32 && R < 64) ?  L >> R : 0;
};
