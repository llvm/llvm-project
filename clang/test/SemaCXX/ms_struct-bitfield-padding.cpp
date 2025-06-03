
// RUN: %clang_cc1 -fsyntax-only -Wms-bitfield-padding -verify -triple armv8 -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -DMS_BITFIELDS -mms-bitfields -verify=msbitfields -triple armv8-apple-macos10.15 -std=c++23 %s

// msbitfields-no-diagnostics

enum Enum1 { Enum1_A, Enum1_B };
enum Enum2 { Enum2_A, Enum2_B };

enum class EnumU32_1 : unsigned { A, B };
enum class EnumU32_2 : unsigned { A, B };
enum class EnumU64 : unsigned long long { A, B };
enum class EnumI32 : int { A, B };
enum class EnumU8 : unsigned char { A, B };
enum class EnumI8 : char { A, B };
enum class EnumU16 : unsigned short { A, B };
enum class EnumI16 : short { A, B };

struct A {
  unsigned int a : 15;
  unsigned int b : 15;
};
static_assert(sizeof(A) == 4);

struct B {
  unsigned int a : 15;
           int b : 15;
};
static_assert(sizeof(B) == 4);

struct C {
  unsigned int a : 15;
           int b : 15;
};
static_assert(sizeof(C) == 4);

struct D {
  Enum1 a : 15;
  Enum1 b : 15;
};
static_assert(sizeof(D) == 4);

struct E {
  Enum1 a : 15;
  Enum2 b : 15;
};
static_assert(sizeof(E) == 4);

struct F {
  EnumU32_1 a : 15;
  EnumU32_2 b : 15;
};
static_assert(sizeof(F) == 4);

struct G {
  EnumU32_1 a : 15;
  EnumU64 b : 15;
  // expected-warning@-1 {{bit-field 'b' of type 'EnumU64' has a different storage size than the preceding bit-field (8 vs 4 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'a' declared here with type 'EnumU32_1'}}
};

#ifdef MS_BITFIELDS
  static_assert(sizeof(G) == 16);
#else
  static_assert(sizeof(G) == 8);
#endif

struct H {
  EnumU32_1 a : 10;
  EnumI32 b : 10;
  EnumU32_1 c : 10;
};
static_assert(sizeof(H) == 4);

struct I {
  EnumU8 a : 3;
  EnumI8 b : 5;
  EnumU32_1 c : 10;
  // expected-warning@-1 {{bit-field 'c' of type 'EnumU32_1' has a different storage size than the preceding bit-field (4 vs 1 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'b' declared here with type 'EnumI8'}}
};
#ifdef MS_BITFIELDS
static_assert(sizeof(I) == 8);
#else
static_assert(sizeof(I) == 4);
#endif

struct J {
  EnumU8 : 0;
  EnumU8 b : 4;
};
static_assert(sizeof(J) == 1);

struct K {
  EnumU8 a : 4;
  EnumU8 : 0;
};
static_assert(sizeof(K) == 1);

struct L {
  EnumU32_1 a : 10;
  EnumU32_2 b : 10;
  EnumU32_1 c : 10;
};

static_assert(sizeof(L) == 4);

struct M {
  EnumU32_1 a : 10;
  EnumI32 b : 10;
  EnumU32_1 c : 10;
};

static_assert(sizeof(M) == 4);

struct N {
  EnumU32_1 a : 10;
  EnumU64 b : 10;
  // expected-warning@-1 {{bit-field 'b' of type 'EnumU64' has a different storage size than the preceding bit-field (8 vs 4 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'a' declared here with type 'EnumU32_1'}}
  EnumU32_1 c : 10;
  // expected-warning@-1 {{bit-field 'c' of type 'EnumU32_1' has a different storage size than the preceding bit-field (4 vs 8 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-5 {{preceding bit-field 'b' declared here with type 'EnumU64'}}
};

#ifdef MS_BITFIELDS
static_assert(sizeof(N) == 24);
#else
static_assert(sizeof(N) == 8);
#endif

struct O {
  EnumU16 a : 10;
  EnumU32_1 b : 10;
  // expected-warning@-1 {{bit-field 'b' of type 'EnumU32_1' has a different storage size than the preceding bit-field (4 vs 2 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'a' declared here with type 'EnumU16'}}
};
#ifdef MS_BITFIELDS
static_assert(sizeof(O) == 8);
#else
static_assert(sizeof(O) == 4);
#endif

struct P {
  EnumU32_1 a : 10;
  EnumU16 b : 10;
  // expected-warning@-1 {{bit-field 'b' of type 'EnumU16' has a different storage size than the preceding bit-field (2 vs 4 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'a' declared here with type 'EnumU32_1'}}
};
#ifdef MS_BITFIELDS
static_assert(sizeof(P) == 8);
#else
static_assert(sizeof(P) == 4);
#endif

struct Q {
  EnumU8 a : 6;
  EnumU16 b : 6;
  // expected-warning@-1 {{bit-field 'b' of type 'EnumU16' has a different storage size than the preceding bit-field (2 vs 1 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'a' declared here with type 'EnumU8'}}
};
#ifdef MS_BITFIELDS
static_assert(sizeof(Q) == 4);
#else
static_assert(sizeof(Q) == 2);
#endif

struct R {
  EnumU16 a : 9;
  EnumU16 b : 9;
  EnumU8 c : 6;
  // expected-warning@-1 {{bit-field 'c' of type 'EnumU8' has a different storage size than the preceding bit-field (1 vs 2 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'b' declared here with type 'EnumU16'}}
};

#ifdef MS_BITFIELDS
static_assert(sizeof(R) == 6);
#else
static_assert(sizeof(R) == 4);
#endif

struct S {
  char a : 4;
  char b : 4;
  char c : 4;
  char d : 4;
  short x : 7;
  // expected-warning@-1 {{bit-field 'x' of type 'short' has a different storage size than the preceding bit-field (2 vs 1 bytes) and will not be packed under the Microsoft ABI}}
  // expected-note@-3 {{preceding bit-field 'd' declared here with type 'char'}}
  // This is a false positive. Reporting this correctly requires duplicating the record layout process
  // in target and MS layout modes, and it's also unclear if that's the correct choice for users of
  // this diagnostic.
  short y : 9;
};

static_assert(sizeof(S) == 4);
