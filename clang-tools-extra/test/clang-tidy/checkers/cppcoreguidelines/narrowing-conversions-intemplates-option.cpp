// RUN: %check_clang_tidy -check-suffix=DEFAULT %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: [ \
// RUN: ]}'

// RUN: %check_clang_tidy -check-suffix=WARN %s \
// RUN: cppcoreguidelines-narrowing-conversions %t -- \
// RUN: -config='{CheckOptions: [ \
// RUN:   {key: cppcoreguidelines-narrowing-conversions.WarnWithinTemplateInstantiation, value: 1} \
// RUN: ]}'

template <typename OrigType>
void assign_in_template(OrigType jj) {
  int ii;
  ii = jj;
  // DEFAULT: Warning disabled because WarnWithinTemplateInstantiation=0.
  // CHECK-MESSAGES-WARN: :[[@LINE-2]]:8: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}

void narrow_inside_template_not_ok() {
  long long j = 123;
  assign_in_template(j);
}

void assign_outside_template(long long jj) {
  int ii;
  ii = jj;
  // CHECK-MESSAGES-DEFAULT: :[[@LINE-1]]:8: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
  // CHECK-MESSAGES-WARN: :[[@LINE-2]]:8: warning: narrowing conversion from 'long long' to signed type 'int' is implementation-defined [cppcoreguidelines-narrowing-conversions]
}

void narrow_outside_template_not_ok() {
  long long j = 123;
  assign_outside_template(j);
}
