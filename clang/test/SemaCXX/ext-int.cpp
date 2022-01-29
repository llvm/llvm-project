// RUN: %clang_cc1 -fsyntax-only -verify %s -Wimplicit-int-conversion -Wno-unused -Wunevaluated-expression -triple x86_64-gnu-linux

template<int Bounds>
struct HasExtInt {
  _BitInt(Bounds) b;
  unsigned _BitInt(Bounds) b2;
};

// Delcaring variables:
_BitInt(33) Declarations(_BitInt(48) &Param) { // Useable in params and returns.
  short _BitInt(43) a; // expected-error {{'short _BitInt' is invalid}}
  _BitInt(43) long b;  // expected-error {{'long _BitInt' is invalid}}

  // These should all be fine:
  const _BitInt(5) c = 3;
  const unsigned _BitInt(5) d; // expected-error {{default initialization of an object of const type 'const unsigned _BitInt(5)'}}
  unsigned _BitInt(5) e = 5;
  _BitInt(5) unsigned f;

  _BitInt(-3) g; // expected-error{{signed _BitInt must have a bit size of at least 2}}
  _BitInt(0) h; // expected-error{{signed _BitInt must have a bit size of at least 2}}
  _BitInt(1) i; // expected-error{{signed _BitInt must have a bit size of at least 2}}
  _BitInt(2) j;;
  unsigned _BitInt(0) k;// expected-error{{unsigned _BitInt must have a bit size of at least 1}}
  unsigned _BitInt(1) l;
  signed _BitInt(1) m; // expected-error{{signed _BitInt must have a bit size of at least 2}}

  constexpr _BitInt(6) n = 33; // expected-warning{{implicit conversion from 'int' to 'const _BitInt(6)' changes value from 33 to -31}}
  constexpr _BitInt(7) o = 33;

  // Check LLVM imposed max size.
  _BitInt(8388609) p;               // expected-error {{signed _BitInt of bit sizes greater than 8388608 not supported}}
  unsigned _BitInt(0xFFFFFFFFFF) q; // expected-error {{unsigned _BitInt of bit sizes greater than 8388608 not supported}}

// Ensure template params are instantiated correctly.
  // expected-error@5{{signed _BitInt must have a bit size of at least 2}}
  // expected-error@6{{unsigned _BitInt must have a bit size of at least 1}}
  // expected-note@+1{{in instantiation of template class }}
  HasExtInt<-1> r;
  // expected-error@5{{signed _BitInt must have a bit size of at least 2}}
  // expected-error@6{{unsigned _BitInt must have a bit size of at least 1}}
  // expected-note@+1{{in instantiation of template class }}
  HasExtInt<0> s;
  // expected-error@5{{signed _BitInt must have a bit size of at least 2}}
  // expected-note@+1{{in instantiation of template class }}
  HasExtInt<1> t;
  HasExtInt<2> u;

  _BitInt(-3.0) v; // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}
  _BitInt(3.0) x; // expected-error {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}

  return 0;
}

template <_BitInt(5) I>
struct ExtIntTemplParam {
  static constexpr _BitInt(5) Var = I;
};

template<typename T>
void deduced_whole_type(T){}
template<int I>
void deduced_bound(_BitInt(I)){}

// Ensure ext-int can be used in template places.
void Templates() {
  ExtIntTemplParam<13> a;
  constexpr _BitInt(3) b = 1;
  ExtIntTemplParam<b> c;
  constexpr _BitInt(9) d = 1;
  ExtIntTemplParam<b> e;

  deduced_whole_type(b);
  deduced_bound(b);
}

template <typename T, typename U>
struct is_same {
  static constexpr bool value = false;
};
template <typename T>
struct is_same<T,T> {
  static constexpr bool value = true;
};

// Reject vector types:
// expected-error@+1{{invalid vector element type '_BitInt(32)'}}
typedef _BitInt(32) __attribute__((vector_size(16))) VecTy;

// Allow _Complex:
_Complex _BitInt(3) Cmplx;

// Reject cases of _Atomic:
// expected-error@+1{{_Atomic cannot be applied to integer type '_BitInt(4)'}}
_Atomic _BitInt(4) TooSmallAtomic;
// expected-error@+1{{_Atomic cannot be applied to integer type '_BitInt(9)'}}
_Atomic _BitInt(9) NotPow2Atomic;
// expected-error@+1{{_Atomic cannot be applied to integer type '_BitInt(128)'}}
_Atomic _BitInt(128) JustRightAtomic;

// Test result types of Unary/Bitwise/Binary Operations:
void Ops() {
  _BitInt(43) x43_s = 1, y43_s = 1;
  _BitInt(sizeof(int) * 8) x32_s = 1, y32_s = 1;
  unsigned _BitInt(sizeof(unsigned) * 8) x32_u = 1, y32_u = 1;
  _BitInt(4) x4_s = 1, y4_s = 1;
  unsigned _BitInt(43) x43_u = 1, y43_u = 1;
  unsigned _BitInt(4) x4_u = 1, y4_u = 1;
  int x_int = 1, y_int = 1;
  unsigned x_uint = 1, y_uint = 1;
  bool b;

  // Signed/unsigned mixed.
  x43_u + y43_s;
  x4_s - y4_u;
  x43_s * y43_u;
  x4_u / y4_s;

  // Different Sizes.
  x43_s + y4_s;
  x43_s - y4_u;
  x43_u * y4_u;
  x4_u / y43_u;

  // Mixed with standard types.
  x43_s + x_int;
  x43_u - x_int;
  x32_s * x_int;
  x32_u / x_int;
  x32_s * x_uint;
  x32_u / x_uint;
  x4_s + x_int;
  x4_u - x_int;
  x4_s + b;
  x4_u - b;
  x43_s + b;
  x43_u - b;
  static_assert(is_same<decltype(x43_s + x_int), _BitInt(43)>::value, "");
  static_assert(is_same<decltype(x43_u + x_int), unsigned _BitInt(43)>::value, "");
  static_assert(is_same<decltype(x32_s + x_int), int>::value, "");
  static_assert(is_same<decltype(x32_u + x_int), unsigned int>::value, "");
  static_assert(is_same<decltype(x32_s + x_uint), unsigned int>::value, "");
  static_assert(is_same<decltype(x32_u + x_uint), unsigned int>::value, "");
  static_assert(is_same<decltype(x4_s + x_int), int>::value, "");
  static_assert(is_same<decltype(x4_u + x_int), int>::value, "");
  static_assert(is_same<decltype(x4_s + x_uint), unsigned int>::value, "");
  static_assert(is_same<decltype(x4_u + x_uint), unsigned int>::value, "");

  // Bitwise checks.
  x43_s % y4_u;
  x43_u % y4_s;
  x4_s | y43_u;
  x4_u | y43_s;

  // compassign.
  x43_s += 33;

  // Comparisons.
  x43_s > 33;
  x4_s > 33; // expected-warning {{result of comparison of constant 33 with expression of type '_BitInt(4)' is always false}}

  // Same size/sign ops don't change type.
  static_assert(is_same<decltype(x43_s + y43_s), _BitInt(43)>::value,"");
  static_assert(is_same<decltype(x4_s - y4_s), _BitInt(4)>::value,"");
  static_assert(is_same<decltype(x43_u * y43_u), unsigned _BitInt(43)>::value,"");
  static_assert(is_same<decltype(x4_u / y4_u), unsigned _BitInt(4)>::value,"");

  // Unary ops shouldn't go through integer promotions.
  static_assert(is_same<decltype(~x43_s), _BitInt(43)>::value,"");
  static_assert(is_same<decltype(~x4_s), _BitInt(4)>::value,"");
  static_assert(is_same<decltype(+x43_s), _BitInt(43)>::value,"");
  static_assert(is_same<decltype(+x4_s), _BitInt(4)>::value,"");
  static_assert(is_same<decltype(-x43_u), unsigned _BitInt(43)>::value,"");
  static_assert(is_same<decltype(-x4_u), unsigned _BitInt(4)>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(++x43_s), _BitInt(43)&>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(--x4_s), _BitInt(4)&>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(x43_s--), _BitInt(43)>::value,"");
  // expected-warning@+1{{expression with side effects has no effect in an unevaluated context}}
  static_assert(is_same<decltype(x4_s++), _BitInt(4)>::value,"");
  static_assert(is_same<decltype(x4_s >> 1), _BitInt(4)>::value,"");
  static_assert(is_same<decltype(x4_u << 1), unsigned _BitInt(4)>::value,"");

  static_assert(sizeof(x43_s) == 8, "");
  static_assert(sizeof(x4_s) == 1, "");

  static_assert(sizeof(_BitInt(3340)) == 424, ""); // 424 * 8 == 3392.
  static_assert(sizeof(_BitInt(1049)) == 136, ""); // 136  *  8 == 1088.

  static_assert(alignof(decltype(x43_s)) == 8, "");
  static_assert(alignof(decltype(x4_s)) == 1, "");

  static_assert(alignof(_BitInt(3340)) == 8, "");
  static_assert(alignof(_BitInt(1049)) == 8, "");
}

constexpr int func() { return 42;}

void ConstexprBitsize() {
  _BitInt(func()) F;
  static_assert(is_same<decltype(F), _BitInt(42)>::value, "");
}

// Useable as an underlying type.
enum AsEnumUnderlyingType : _BitInt(33) {
};

void overloaded(int);
void overloaded(_BitInt(32));
void overloaded(_BitInt(33));
void overloaded(short);
//expected-note@+1{{candidate function}}
void overloaded2(_BitInt(32));
//expected-note@+1{{candidate function}}
void overloaded2(_BitInt(33));
//expected-note@+1{{candidate function}}
void overloaded2(short);

void overload_use() {
  int i;
  _BitInt(32) i32;
  _BitInt(33) i33;
  short s;

  // All of these get their corresponding exact matches.
  overloaded(i);
  overloaded(i32);
  overloaded(i33);
  overloaded(s);

  overloaded2(i); // expected-error{{call to 'overloaded2' is ambiguous}}

  overloaded2(i32);

  overloaded2(s);
}

// no errors expected, this should 'just work'.
struct UsedAsBitField {
  _BitInt(3) F : 3;
  _BitInt(3) G : 3;
  _BitInt(3) H : 3;
};

struct CursedBitField {
  _BitInt(4) A : 8; // expected-warning {{width of bit-field 'A' (8 bits) exceeds the width of its type; value will be truncated to 4 bits}}
};

// expected-error@+1{{mode attribute only supported for integer and floating-point types}}
typedef _BitInt(33) IllegalMode __attribute__((mode(DI)));

void ImplicitCasts(_BitInt(31) s31, _BitInt(33) s33, int i) {
  // expected-warning@+1{{implicit conversion loses integer precision}}
  s31 = i;
  // expected-warning@+1{{implicit conversion loses integer precision}}
  s31 = s33;
  s33 = i;
  s33 = s31;
  i = s31;
  // expected-warning@+1{{implicit conversion loses integer precision}}
  i = s33;
}

void Ternary(_BitInt(30) s30, _BitInt(31) s31a, _BitInt(31) s31b,
             _BitInt(32) s32, bool b) {
  b ? s30 : s31a;
  b ? s31a : s30;
  b ? s32 : (int)0;
  (void)(b ? s31a : s31b);
  (void)(s30 ? s31a : s31b);

  static_assert(is_same<decltype(b ? s30 : s31a), _BitInt(31)>::value, "");
  static_assert(is_same<decltype(b ? s32 : s30), _BitInt(32)>::value, "");
  static_assert(is_same<decltype(b ? s30 : 0), int>::value, "");
}

void FromPaper1() {
  // Test the examples of conversion and promotion rules from C2x 6.3.1.8.
  _BitInt(2) a2 = 1;
  _BitInt(3) a3 = 2;
  _BitInt(33) a33 = 1;
  char c = 3;

  static_assert(is_same<decltype(a2 * a3), _BitInt(3)>::value, "");
  static_assert(is_same<decltype(a2 * c), int>::value, "");
  static_assert(is_same<decltype(a33 * c), _BitInt(33)>::value, "");
}

void FromPaper2(_BitInt(8) a1, _BitInt(24) a2) {
  static_assert(is_same<decltype(a1 * (_BitInt(32))a2), _BitInt(32)>::value, "");
}
