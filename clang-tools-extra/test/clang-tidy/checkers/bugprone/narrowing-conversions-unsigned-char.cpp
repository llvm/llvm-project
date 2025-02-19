// RUN: %check_clang_tidy %s bugprone-narrowing-conversions %t \
// RUN: -- -- -target x86_64-unknown-linux -funsigned-char

void narrow_integer_to_unsigned_integer_is_ok() {
  signed char sc;
  short s;
  int i;
  long l;
  long long ll;

  char c;
  unsigned short us;
  unsigned int ui;
  unsigned long ul;
  unsigned long long ull;

  ui = sc;
  c = s;
  c = i;
  c = l;
  c = ll;

  c = c;
  c = us;
  c = ui;
  c = ul;
  c = ull;
}

void narrow_integer_to_signed_integer_is_not_ok() {
  signed char sc;
  short s;
  int i;
  long l;
  long long ll;

  char c;
  unsigned short us;
  unsigned int ui;
  unsigned long ul;
  unsigned long long ull;

  sc = sc;
  sc = s;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'short' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
  sc = i;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'int' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
  sc = l;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'long' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
  sc = ll;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'long long' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]

  sc = c;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'char' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
  sc = us;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'unsigned short' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
  sc = ui;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'unsigned int' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
  sc = ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'unsigned long' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
  sc = ull;
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: narrowing conversion from 'unsigned long long' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
}

void narrow_constant_to_unsigned_integer_is_ok() {
  char c1 = -128; // unsigned dst type is well defined.
  char c2 = 127;  // unsigned dst type is well defined.
  char c3 = -129; // unsigned dst type is well defined.
  char c4 = 128;  // unsigned dst type is well defined.
  unsigned char uc1 = 0;
  unsigned char uc2 = 255;
  unsigned char uc3 = -1;  // unsigned dst type is well defined.
  unsigned char uc4 = 256; // unsigned dst type is well defined.
  signed char sc = 128;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: narrowing conversion from constant value 128 (0x00000080) of type 'int' to signed type 'signed char' is implementation-defined [bugprone-narrowing-conversions]
}

void narrow_conditional_operator_contant_to_unsigned_is_ok(bool b) {
  // conversion to unsigned char type is well defined.
  char c1 = b ? 1 : 0;
  char c2 = b ? 1 : 256;
  char c3 = b ? -1 : 0;
}
