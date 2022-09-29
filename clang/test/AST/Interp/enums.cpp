// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

enum class EC : short {
  A, B, C
};
static_assert(static_cast<int>(EC::A) == 0, "");
static_assert(static_cast<int>(EC::B) == 1, "");
static_assert(static_cast<int>(EC::C) == 2, "");
static_assert(sizeof(EC) == sizeof(short), "");

constexpr EC ec = EC::C;
static_assert(static_cast<int>(ec) == 2, "");

constexpr int N = 12;
constexpr int M = 2;

enum CE {
  ONE = -1,
  TWO = 2,
  THREE,
  FOUR = 4,
  FIVE = N + M,
  SIX = FIVE + 2,
  MAX = __INT_MAX__ * 2U + 1U
};
static_assert(ONE == -1, "");
static_assert(THREE == 3, "");
static_assert(FIVE == 14, "");
static_assert(SIX == 16, "");

constexpr EC testEnums() {
  EC e = EC::C;

  e = EC::B;

  EC::B = e; // expected-error{{expression is not assignable}} \
             // ref-error{{expression is not assignable}}

  return e;
}

constexpr EC getB() {
  EC e = EC::C;
  e = EC::B;
  return e;
}


static_assert(getB() == EC::B, "");


enum E { // expected-warning{{enumeration values exceed range of largest integer}} \
         // ref-warning{{enumeration values exceed range of largest integer}}
  E1 = -__LONG_MAX__ -1L,
  E2 = __LONG_MAX__ *2UL+1UL
};
