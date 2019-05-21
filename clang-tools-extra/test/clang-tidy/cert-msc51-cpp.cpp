// RUN: %check_clang_tidy %s cert-msc51-cpp %t -- \
// RUN:     -config="{CheckOptions: [{key: cert-msc51-cpp.DisallowedSeedTypes, value: 'some_type,time_t'}]}"

namespace std {

void srand(int seed);

template <class UIntType, UIntType a, UIntType c, UIntType m>
struct linear_congruential_engine {
  linear_congruential_engine(int _ = 0);
  void seed(int _ = 0);
};
using default_random_engine = linear_congruential_engine<unsigned int, 1, 2, 3>;

using size_t = int;
template <class UIntType, size_t w, size_t n, size_t m, size_t r,
          UIntType a, size_t u, UIntType d, size_t s,
          UIntType b, size_t t,
          UIntType c, size_t l, UIntType f>
struct mersenne_twister_engine {
  mersenne_twister_engine(int _ = 0);
  void seed(int _ = 0);
};
using mt19937 = mersenne_twister_engine<unsigned int, 32, 624, 397, 21, 0x9908b0df, 11, 0xffffffff, 7, 0x9d2c5680, 15, 0xefc60000, 18, 1812433253>;

template <class UIntType, size_t w, size_t s, size_t r>
struct subtract_with_carry_engine {
  subtract_with_carry_engine(int _ = 0);
  void seed(int _ = 0);
};
using ranlux24_base = subtract_with_carry_engine<unsigned int, 24, 10, 24>;

template <class Engine, size_t p, size_t r>
struct discard_block_engine {
  discard_block_engine();
  discard_block_engine(int _);
  void seed();
  void seed(int _);
};
using ranlux24 = discard_block_engine<ranlux24_base, 223, 23>;

template <class Engine, size_t w, class UIntType>
struct independent_bits_engine {
  independent_bits_engine();
  independent_bits_engine(int _);
  void seed();
  void seed(int _);
};
using independent_bits = independent_bits_engine<ranlux24_base, 223, int>;

template <class Engine, size_t k>
struct shuffle_order_engine {
  shuffle_order_engine();
  shuffle_order_engine(int _);
  void seed();
  void seed(int _);
};
using shuffle_order = shuffle_order_engine<ranlux24_base, 223>;

struct random_device {
  random_device();
  int operator()();
};
} // namespace std

using time_t = unsigned int;
time_t time(time_t *t);

void f() {
  const int seed = 2;
  time_t t;

  std::srand(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::srand(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::srand(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]

  // One instantiation for every engine
  std::default_random_engine engine1;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  std::default_random_engine engine2(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::default_random_engine engine3(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::default_random_engine engine4(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]
  engine1.seed();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  engine1.seed(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine1.seed(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine1.seed(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]

  std::mt19937 engine5;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  std::mt19937 engine6(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::mt19937 engine7(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::mt19937 engine8(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]
  engine5.seed();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  engine5.seed(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine5.seed(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine5.seed(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]

  std::ranlux24_base engine9;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  std::ranlux24_base engine10(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::ranlux24_base engine11(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::ranlux24_base engine12(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]
  engine9.seed();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  engine9.seed(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine9.seed(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine9.seed(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]

  std::ranlux24 engine13;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  std::ranlux24 engine14(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::ranlux24 engine15(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::ranlux24 engine16(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]
  engine13.seed();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  engine13.seed(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine13.seed(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine13.seed(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]

  std::independent_bits engine17;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  std::independent_bits engine18(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::independent_bits engine19(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::independent_bits engine20(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]
  engine17.seed();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  engine17.seed(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine17.seed(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine17.seed(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]

  std::shuffle_order engine21;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  std::shuffle_order engine22(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::shuffle_order engine23(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  std::shuffle_order engine24(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]
  engine21.seed();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a default argument will generate a predictable sequence of values [cert-msc51-cpp]
  engine21.seed(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine21.seed(seed);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a constant value will generate a predictable sequence of values [cert-msc51-cpp]
  engine21.seed(time(&t));
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: random number generator seeded with a disallowed source of seed value will generate a predictable sequence of values [cert-msc51-cpp]
}

struct A {
  A(int _ = 0);
  void seed(int _ = 0);
};

void g() {
  int n = 1;
  std::default_random_engine engine1(n);
  std::mt19937 engine2(n);
  std::ranlux24_base engine3(n);
  std::ranlux24 engine4(n);
  std::independent_bits engine5(n);
  std::shuffle_order engine6(n);

  std::random_device dev;
  std::default_random_engine engine7(dev());
  std::mt19937 engine8(dev());
  std::ranlux24_base engine9(dev());
  std::ranlux24 engine10(dev());
  std::independent_bits engine11(dev());
  std::shuffle_order engine12(dev());

  A a1;
  A a2(1);
  a1.seed();
  a1.seed(1);
  a1.seed(n);
}
