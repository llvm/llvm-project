// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fms-extensions -std=c++11 -verify %s
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fms-extensions -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++11 -fms-extensions -verify=ref %s
// RUN: %clang_cc1 -std=c++20 -fms-extensions -verify=ref %s


using MaxBitInt = _BitInt(128);
#define INT_MIN (~__INT_MAX__)

constexpr _BitInt(2) A = 0;
constexpr _BitInt(2) B = A + 1;
constexpr _BitInt(2) C = B + 1; // expected-warning {{from 2 to -2}} \
                                // ref-warning {{from 2 to -2}}
static_assert(C == -2, "");
static_assert(C - B == A, ""); // expected-error {{not an integral constant expression}} \
                               // expected-note {{value -3 is outside the range of representable values}} \
                               // ref-error {{not an integral constant expression}} \
                               // ref-note {{value -3 is outside the range of representable values}}

static_assert(B - 1 == 0, "");

constexpr MaxBitInt A_ = 0;
constexpr MaxBitInt B_ = A_ + 1;
static_assert(B_ == 1, "");

constexpr MaxBitInt BitIntZero{};
static_assert(BitIntZero == 0, "");
constexpr unsigned _BitInt(128) UBitIntZero{};
static_assert(UBitIntZero == 0, "");

constexpr _BitInt(2) BitIntZero2{};
static_assert(BitIntZero2 == 0, "");
constexpr unsigned _BitInt(1) UBitIntZero1{};
static_assert(UBitIntZero1 == 0, "");

constexpr unsigned _BitInt(2) BI1 = 3u;
static_assert(BI1 == 3, "");

constexpr _BitInt(4) MulA = 5;
constexpr _BitInt(4) MulB = 7;
static_assert(MulA * MulB == 50, ""); // ref-error {{not an integral constant expression}} \
                                      // ref-note {{value 35 is outside the range of representable values of type '_BitInt(4)'}} \
                                      // expected-error {{not an integral constant expression}} \
                                      // expected-note {{value 35 is outside the range of representable values of type '_BitInt(4)'}}
static_assert(MulA * 5 == 25, "");
static_assert(-1 * MulB == -7, "");


constexpr _BitInt(4) DivA = 2;
constexpr _BitInt(2) DivB = 1;
static_assert(DivA / DivB == 2, "");

constexpr _BitInt(4) DivC = DivA / 0; // ref-error {{must be initialized by a constant expression}} \
                                      // ref-note {{division by zero}} \
                                      // expected-error {{must be initialized by a constant expression}} \
                                      // expected-note {{division by zero}}

constexpr _BitInt(7) RemA = 47;
constexpr _BitInt(6) RemB = 9;
static_assert(RemA % RemB == 2, "");
static_assert(RemA % 0 == 1, ""); // ref-error {{not an integral constant expression}} \
                                  // ref-note {{division by zero}} \
                                  // expected-error {{not an integral constant expression}} \
                                  // expected-note {{division by zero}}

constexpr _BitInt(32) bottom = -1;
constexpr _BitInt(32) top = INT_MIN;
constexpr _BitInt(32) nope = top / bottom;  // ref-error {{must be initialized by a constant expression}} \
                                            // ref-note {{value 2147483648 is outside the range}} \
                                            // expected-error {{must be initialized by a constant expression}} \
                                            // expected-note {{value 2147483648 is outside the range}}
constexpr _BitInt(32) noooo = top % bottom; // ref-error {{must be initialized by a constant expression}} \
                                            // ref-note {{value 2147483648 is outside the range}} \
                                            // expected-error {{must be initialized by a constant expression}} \
                                            // expected-note {{value 2147483648 is outside the range}}

namespace APCast {
  constexpr _BitInt(10) A = 1;
  constexpr _BitInt(11) B = A;
  static_assert(B == 1, "");
  constexpr _BitInt(16) B2 = A;
  static_assert(B2 == 1, "");
  constexpr _BitInt(32) B3 = A;
  static_assert(B3 == 1, "");
  constexpr unsigned _BitInt(32) B4 = A;
  static_assert(B4 == 1, "");
}

#ifdef __SIZEOF_INT128__
typedef __int128 int128_t;
typedef unsigned __int128 uint128_t;
static const __uint128_t UINT128_MAX =__uint128_t(__int128_t(-1L));
static_assert(UINT128_MAX == -1, "");
static_assert(UINT128_MAX == 1, ""); // expected-error {{static assertion failed}} \
                                     // expected-note {{'340282366920938463463374607431768211455 == 1'}} \
                                     // ref-error {{static assertion failed}} \
                                     // ref-note {{'340282366920938463463374607431768211455 == 1'}}

static const __int128_t INT128_MAX = UINT128_MAX >> (__int128_t)1;
static_assert(INT128_MAX != 0, "");
static_assert(INT128_MAX == 0, ""); // expected-error {{failed}} \
                                    // expected-note {{evaluates to '170141183460469231731687303715884105727 == 0'}} \
                                    // ref-error {{failed}} \
                                    // ref-note {{evaluates to '170141183460469231731687303715884105727 == 0'}}
static const __int128_t INT128_MIN = -INT128_MAX - 1;

namespace i128 {

  constexpr int128_t I128_1 = 12;
  static_assert(I128_1 == 12, "");
  static_assert(I128_1 != 10, "");
  static_assert(I128_1 != 12, ""); // expected-error{{failed}} \
                                   // ref-error{{failed}} \
                                   // expected-note{{evaluates to}} \
                                   // ref-note{{evaluates to}}

  static const __uint128_t UINT128_MAX =__uint128_t(__int128_t(-1L));
  static_assert(UINT128_MAX == -1, "");
  static_assert(UINT128_MAX == 1, ""); // expected-error {{static assertion failed}} \
                                       // expected-note {{'340282366920938463463374607431768211455 == 1'}} \
                                       // ref-error {{static assertion failed}} \
                                       // ref-note {{'340282366920938463463374607431768211455 == 1'}}

  constexpr uint128_t TooMuch = UINT128_MAX * 2;

  static const __int128_t INT128_MAX = UINT128_MAX >> (__int128_t)1;
  static_assert(INT128_MAX != 0, "");
  static_assert(INT128_MAX == 0, ""); // expected-error {{failed}} \
                                      // expected-note {{evaluates to '170141183460469231731687303715884105727 == 0'}} \
                                      // ref-error {{failed}} \
                                      // ref-note {{evaluates to '170141183460469231731687303715884105727 == 0'}}

  constexpr int128_t TooMuch2 = INT128_MAX * INT128_MAX; // ref-error {{must be initialized by a constant expression}} \
                                                // ref-note {{value 28948022309329048855892746252171976962977213799489202546401021394546514198529 is outside the range of representable}} \
                                                // expected-error {{must be initialized by a constant expression}} \
                                                // expected-note {{value 28948022309329048855892746252171976962977213799489202546401021394546514198529 is outside the range of representable}}

  static const __int128_t INT128_MIN = -INT128_MAX - 1;
  constexpr __int128 A = INT128_MAX + 1; // expected-error {{must be initialized by a constant expression}} \
                                         // expected-note {{value 170141183460469231731687303715884105728 is outside the range}} \
                                         // ref-error {{must be initialized by a constant expression}} \
                                         // ref-note {{value 170141183460469231731687303715884105728 is outside the range}}
  constexpr int128_t Two = (int128_t)1 << 1ul;
  static_assert(Two == 2, "");
  static_assert(Two, "");
  constexpr bool CastedToBool = Two;
  static_assert(CastedToBool, "");

  constexpr uint128_t AllOnes = ~static_cast<uint128_t>(0);
  static_assert(AllOnes == UINT128_MAX, "");

  constexpr uint128_t i128Zero{};
  static_assert(i128Zero == 0, "");
  constexpr uint128_t ui128Zero{};
  static_assert(ui128Zero == 0, "");

#if __cplusplus >= 201402L
  template <typename T>
  constexpr T CastFrom(__int128_t A) {
    T B = (T)A;
    return B;
  }
  static_assert(CastFrom<char>(12) == 12, "");
  static_assert(CastFrom<unsigned char>(12) == 12, "");
  static_assert(CastFrom<long>(12) == 12, "");
  static_assert(CastFrom<unsigned short>(12) == 12, "");
  static_assert(CastFrom<int128_t>(12) == 12, "");
  static_assert(CastFrom<float>(12) == 12, "");
  static_assert(CastFrom<double>(12) == 12, "");
  static_assert(CastFrom<long double>(12) == 12, "");

  static_assert(CastFrom<char>(AllOnes) == -1, "");
  static_assert(CastFrom<unsigned char>(AllOnes) == 0xFF, "");
  static_assert(CastFrom<long>(AllOnes) == -1, "");
  static_assert(CastFrom<unsigned short>(AllOnes) == 0xFFFF, "");
  static_assert(CastFrom<int>(AllOnes) == -1, "");
  static_assert(CastFrom<int128_t>(AllOnes) == -1, "");
  static_assert(CastFrom<uint128_t>(AllOnes) == AllOnes, "");

  template <typename T>
  constexpr __int128 CastTo(T A) {
    int128_t B = (int128_t)A;
    return B;
  }
  static_assert(CastTo<char>(12) == 12, "");
  static_assert(CastTo<unsigned char>(12) == 12, "");
  static_assert(CastTo<long>(12) == 12, "");
  static_assert(CastTo<unsigned long long>(12) == 12, "");
  static_assert(CastTo<float>(12) == 12, "");
  static_assert(CastTo<double>(12) == 12, "");
  static_assert(CastTo<long double>(12) == 12, "");
#endif

  constexpr int128_t Error = __LDBL_MAX__; // ref-warning {{implicit conversion of out of range value}} \
                                           // ref-error {{must be initialized by a constant expression}} \
                                           // ref-note {{is outside the range of representable values of type}} \
                                           // expected-warning {{implicit conversion of out of range value}} \
                                           // expected-error {{must be initialized by a constant expression}} \
                                           // expected-note {{is outside the range of representable values of type}}

  constexpr uint128_t Zero = 0;
  static_assert((Zero -1) == -1, "");
  constexpr int128_t Five = 5;
  static_assert(Five - Zero == Five, "");

  constexpr int128_t Sub1 = INT128_MIN - 1; // expected-error {{must be initialized by a constant expression}} \
                                            // expected-note {{-170141183460469231731687303715884105729 is outside the range}} \
                                            // ref-error {{must be initialized by a constant expression}} \
                                            // ref-note {{-170141183460469231731687303715884105729 is outside the range}}
}

namespace AddSubOffset {
  constexpr __int128 A = 1;
  constexpr int arr[] = {1,2,3};
  constexpr const int *P = arr + A;
  static_assert(*P == 2, "");
  constexpr const int *P2 = P - A;
  static_assert(*P2 == 1,"");
}

namespace Bitfields {
  struct S1 {
    unsigned _BitInt(128) a : 2;
  };
  constexpr S1 s1{100}; // ref-warning {{changes value from 100 to 0}} \
                        // expected-warning {{changes value from 100 to 0}}
  constexpr S1 s12{3};
  static_assert(s12.a == 3, "");

  struct S2 {
    unsigned __int128 a : 2;
  };
  constexpr S2 s2{100}; // ref-warning {{changes value from 100 to 0}} \
                        // expected-warning {{changes value from 100 to 0}}
}

namespace BitOps {
  constexpr unsigned __int128 UZero = 0;
  constexpr unsigned __int128 Max = ~UZero;
  static_assert(Max == ~0, "");
  static_assert((Max & 0) == 0, "");
  static_assert((UZero | 0) == 0, "");
  static_assert((Max ^ Max) == 0, "");
  static_assert((Max & 1) == 1, "");
  static_assert((UZero | 1) == 1, "");
  static_assert((Max ^ UZero) == Max, "");
}

namespace IncDec {
#if __cplusplus >= 201402L
  constexpr int128_t maxPlus1(bool Pre) {
    int128_t a = INT128_MAX;

    if (Pre)
      ++a; // ref-note {{value 170141183460469231731687303715884105728 is outside the range}} \
           // expected-note {{value 170141183460469231731687303715884105728 is outside the range}}
    else
      a++; // ref-note {{value 170141183460469231731687303715884105728 is outside the range}} \
           // expected-note {{value 170141183460469231731687303715884105728 is outside the range}}
    return a;
  }
  static_assert(maxPlus1(true) == 0, ""); // ref-error {{not an integral constant expression}} \
                                          // ref-note {{in call to}} \
                                          // expected-error {{not an integral constant expression}} \
                                          // expected-note {{in call to}}
  static_assert(maxPlus1(false) == 0, ""); // ref-error {{not an integral constant expression}} \
                                           // ref-note {{in call to}} \
                                           // expected-error {{not an integral constant expression}} \
                                           // expected-note {{in call to}}

  constexpr int128_t inc1(bool Pre) {
    int128_t A = 0;
    if (Pre)
      ++A;
    else
      A++;
    return A;
  }
  static_assert(inc1(true) == 1, "");
  static_assert(inc1(false) == 1, "");

  constexpr int128_t dec1(bool Pre) {
    int128_t A = 2;
    if (Pre)
      --A;
    else
      A--;
    return A;
  }
  static_assert(dec1(true) == 1, "");
  static_assert(dec1(false) == 1, "");
#endif
}

#endif
