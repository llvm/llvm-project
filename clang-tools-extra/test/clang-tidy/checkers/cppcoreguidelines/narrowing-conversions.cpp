// RUN: %check_clang_tidy %s cppcoreguidelines-narrowing-conversions %t \
// RUN: -config="{CheckOptions: { \
// RUN:   cppcoreguidelines-narrowing-conversions.WarnOnFloatingPointNarrowingConversion: false}}" \
// RUN: -- -target x86_64-unknown-linux -fsigned-char

float ceil(float);
namespace std {
double ceil(double);
long double floor(long double);
} // namespace std

namespace floats {

struct ConvertsToFloat {
  operator float() const { return 0.5f; }
};

float operator"" _float(unsigned long long);

void narrow_fp_to_int_not_ok(double d) {
  int i = 0;
  i = d;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from constant 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = static_cast<float>(d);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = ConvertsToFloat();
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i = 15_float;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += d;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += 0.5;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i *= 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i /= 0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += (double)0.5f;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from constant 'double' to 'int' [cppcoreguidelines-narrowing-conversions]
  i += 2.0;
  i += 2.0f;
}

double operator"" _double(unsigned long long);

float narrow_double_to_float_return() {
  return 0.5; // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.
}

void narrow_double_to_float_ok(double d) {
  float f;
  f = d;
  f = 15_double;
}

void narrow_fp_constants() {
  float f;
  f = 0.5; // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.

  f = __builtin_huge_valf();  // max float is not narrowing.
  f = -__builtin_huge_valf(); // -max float is not narrowing.
  f = __builtin_inff();       // float infinity is not narrowing.
  f = __builtin_nanf("0");    // float NaN is not narrowing.

  f = __builtin_huge_val();  // max double is not within-range of float.
  f = -__builtin_huge_val(); // -max double is not within-range of float.
  f = __builtin_inf();       // double infinity is not within-range of float.
  f = __builtin_nan("0");    // double NaN is not narrowing.
}

void narrow_double_to_float_not_ok_binary_ops(double d) {
  float f;
  f += 0.5;          // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.
  f += 2.0;          // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.
  f *= 0.5;          // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.
  f /= 0.5;          // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.
  f += (double)0.5f; // [dcl.init.list] 7.2 : in-range fp constant to narrower float is not a narrowing.
  f += d;            // We do not warn about floating point narrowing by default.
}

void narrow_fp_constant_to_bool_not_ok() {
  bool b1 = 1.0;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: narrowing conversion from constant 'double' to 'bool' [cppcoreguidelines-narrowing-conversions]
  bool b2 = 1.0f;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: narrowing conversion from constant 'float' to 'bool' [cppcoreguidelines-narrowing-conversions]
}

void narrow_integer_to_floating() {
  {
    long long ll; // 64 bits
    float f = ll; // doesn't fit in 24 bits
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: narrowing conversion from 'long long' to 'float' [cppcoreguidelines-narrowing-conversions]
    double d = ll; // doesn't fit in 53 bits.
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: narrowing conversion from 'long long' to 'double' [cppcoreguidelines-narrowing-conversions]
  }
  {
    int i;       // 32 bits
    float f = i; // doesn't fit in 24 bits
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: narrowing conversion from 'int' to 'float' [cppcoreguidelines-narrowing-conversions]
    double d = i; // fits in 53 bits.
  }
  {
    short n1, n2;
    float f = n1 + n2; // 'n1 + n2' is of type 'int' because of integer rules
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: narrowing conversion from 'int' to 'float' [cppcoreguidelines-narrowing-conversions]
  }
  {
    short s;      // 16 bits
    float f = s;  // fits in 24 bits
    double d = s; // fits in 53 bits.
  }
}

void narrow_integer_to_unsigned_integer_is_ok() {
  char c;
  short s;
  int i;
  long l;
  long long ll;

  unsigned char uc;
  unsigned short us;
  unsigned int ui;
  unsigned long ul;
  unsigned long long ull;

  ui = c;
  uc = s;
  uc = i;
  uc = l;
  uc = ll;

  uc = uc;
  uc = us;
  uc = ui;
  uc = ul;
  uc = ull;
}

void narrow_integer_to_signed_integer_is_not_ok() {
  char c;
  short s;
  int i;
  long l;
  long long ll;

  unsigned char uc;
  unsigned short us;
  unsigned int ui;
  unsigned long ul;
  unsigned long long ull;

  c = c;
  c = s;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'short' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  c = i;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'int' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  c = l;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'long' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  c = ll;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'long long' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]

  c = uc;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned char' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  c = us;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned short' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  c = ui;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned int' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  c = ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned long' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  c = ull;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned long long' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]

  i = c;
  i = s;
  i = i;
  i = l;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  i = ll;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]

  i = uc;
  i = us;
  i = ui;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  i = ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  i = ull;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]

  ll = c;
  ll = s;
  ll = i;
  ll = l;
  ll = ll;

  ll = uc;
  ll = us;
  ll = ui;
  ll = ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'unsigned long' to signed type 'long long' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  ll = ull;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'unsigned long long' to signed type 'long long' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}

void narrow_constant_to_unsigned_integer_is_ok() {
  unsigned char uc1 = 0;
  unsigned char uc2 = 255;
  unsigned char uc3 = -1;  // unsigned dst type is well defined.
  unsigned char uc4 = 256; // unsigned dst type is well defined.
  unsigned short us1 = 0;
  unsigned short us2 = 65535;
  unsigned short us3 = -1;    // unsigned dst type is well defined.
  unsigned short us4 = 65536; // unsigned dst type is well defined.
}

void narrow_constant_to_signed_integer_is_not_ok() {
  char c1 = -128;
  char c2 = 127;
  char c3 = -129;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: narrowing conversion from constant value -129 (0xFFFFFF7F) of type 'int' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  char c4 = 128;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: narrowing conversion from constant value 128 (0x00000080) of type 'int' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]

  short s1 = -32768;
  short s2 = 32767;
  short s3 = -32769;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: narrowing conversion from constant value -32769 (0xFFFF7FFF) of type 'int' to signed type 'short' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  short s4 = 32768;
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: narrowing conversion from constant value 32768 (0x00008000) of type 'int' to signed type 'short' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}

void narrow_conditional_operator_contant_to_unsigned_is_ok(bool b) {
  // conversion to unsigned dst type is well defined.
  unsigned char c1 = b ? 1 : 0;
  unsigned char c2 = b ? 1 : 256;
  unsigned char c3 = b ? -1 : 0;
}

void narrow_conditional_operator_contant_to_signed_is_not_ok(bool b) {
  char uc1 = b ? 1 : 0;
  char uc2 = b ? 1 : 128;
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: narrowing conversion from constant value 128 (0x00000080) of type 'int' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  char uc3 = b ? -129 : 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: narrowing conversion from constant value -129 (0xFFFFFF7F) of type 'int' to signed type 'char' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  unsigned long long ysize;
  long long mirror = b ? -1 : ysize - 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: narrowing conversion from constant value 18446744073709551615 (0xFFFFFFFFFFFFFFFF) of type 'unsigned long long' to signed type 'long long' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // CHECK-MESSAGES: :[[@LINE-2]]:37: warning: narrowing conversion from 'unsigned long long' to signed type 'long long' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}

void narrow_constant_to_floating_point() {
  float f_ok = 1ULL << 24;              // fits in 24 bits mantissa.
  float f_not_ok = (1ULL << 24) + 1ULL; // doesn't fit in 24 bits mantissa.
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: narrowing conversion from constant value 16777217 of type 'unsigned long long' to 'float' [cppcoreguidelines-narrowing-conversions]
  double d_ok = 1ULL << 53;              // fits in 53 bits mantissa.
  double d_not_ok = (1ULL << 53) + 1ULL; // doesn't fit in 53 bits mantissa.
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: narrowing conversion from constant value 9007199254740993 of type 'unsigned long long' to 'double' [cppcoreguidelines-narrowing-conversions]
}

void casting_integer_to_bool_is_ok() {
  int i;
  while (i) {
  }
  for (; i;) {
  }
  if (i) {
  }
}

void casting_float_to_bool_is_not_ok() {
  float f;
  while (f) {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: narrowing conversion from 'float' to 'bool' [cppcoreguidelines-narrowing-conversions]
  }
  for (; f;) {
    // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: narrowing conversion from 'float' to 'bool' [cppcoreguidelines-narrowing-conversions]
  }
  if (f) {
    // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'bool' [cppcoreguidelines-narrowing-conversions]
  }
}

void legitimate_comparison_do_not_warn(unsigned long long size) {
  for (int i = 0; i < size; ++i) {
  }
}

void ok(double d) {
  int i = 0;
  i = 1;
  i = static_cast<int>(0.5);
  i = static_cast<int>(d);
  i = std::ceil(0.5);
  i = ::std::floor(0.5);
  {
    using std::ceil;
    i = ceil(0.5f);
  }
  i = ceil(0.5f);
}

void ok_binary_ops(double d) {
  int i = 0;
  i += 1;
  i += static_cast<int>(0.5);
  i += static_cast<int>(d);
  i += (int)d;
  i += std::ceil(0.5);
  i += ::std::floor(0.5);
  {
    using std::ceil;
    i += ceil(0.5f);
  }
  i += ceil(0.5f);
}

// We're bailing out in templates and macros.
template <typename T1, typename T2>
void f(T1 one, T2 two) {
  one += two;
}

void template_context() {
  f(1, 2);
  f(1, .5f);
  f(1, .5);
  f(1, .5l);
}

#define DERP(i, j) (i += j)

void macro_context() {
  int i = 0;
  DERP(i, 2);
  DERP(i, .5f);
  DERP(i, .5);
  DERP(i, .5l);
}

// We understand typedefs.
void typedef_context() {
  typedef long long myint64_t;
  int i;
  myint64_t i64;

  i64 = i64; // Okay, no conversion.
  i64 = i;   // Okay, no narrowing.

  i = i64;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: narrowing conversion from 'myint64_t' (aka 'long long') to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}

} // namespace floats
