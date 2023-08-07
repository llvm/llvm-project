// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- 

// RUN: %check_clang_tidy -check-suffix=DISABLED %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: { \
// RUN:   cppcoreguidelines-narrowing-conversions.WarnOnEquivalentBitWidth: 0}}'

void narrowing_equivalent_bitwidth() {
  int i;
  unsigned int ui;
  i = ui;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // DISABLED: Warning disabled with WarnOnEquivalentBitWidth=0.

  float f;
  i = f;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'float' to 'int' [cppcoreguidelines-narrowing-conversions]
  // DISABLED: Warning disabled with WarnOnEquivalentBitWidth=0.

  f = i;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'int' to 'float' [cppcoreguidelines-narrowing-conversions]
  // DISABLED: Warning disabled with WarnOnEquivalentBitWidth=0.

  long long ll;
  double d;
  ll = d;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:8: warning: narrowing conversion from 'double' to 'long long' [cppcoreguidelines-narrowing-conversions]
  // DISABLED: Warning disabled with WarnOnEquivalentBitWidth=0.

  d = ll;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'long long' to 'double' [cppcoreguidelines-narrowing-conversions]
  // DISABLED: Warning disabled with WarnOnEquivalentBitWidth=0.
}

void most_narrowing_is_not_ok() {
  int i;
  long long ui;
  i = ui;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // CHECK-MESSAGES-DISABLED: :[[@LINE-2]]:7: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}
