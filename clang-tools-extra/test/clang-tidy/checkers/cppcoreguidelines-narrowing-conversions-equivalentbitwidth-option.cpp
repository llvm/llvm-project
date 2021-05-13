// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: [ \
// RUN: ]}'

// RUN: %check_clang_tidy -check-suffix=DISABLED %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: [ \
// RUN:   {key: cppcoreguidelines-narrowing-conversions.WarnOnEquivalentBitWidth, value: 0} \
// RUN: ]}'

void narrowing_equivalent_bitwidth() {
  int i;
  unsigned int j;
  i = j;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'unsigned int' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // DISABLED: Warning disabled with WarnOnEquivalentBitWidth=0.
}

void most_narrowing_is_not_ok() {
  int i;
  long long j;
  i = j;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:7: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // CHECK-MESSAGES-DISABLED: :[[@LINE-2]]:7: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}
