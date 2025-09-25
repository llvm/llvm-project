// RUN: %check_clang_tidy -check-suffix=NON-STRICT-REGEX         %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomFunctions: '::name_match,replacement,is a qualname match;^::prefix_match,,is matched on qualname prefix;^::S::member_match_,,is matched on a C++ class member'}}"
// RUN: %check_clang_tidy -check-suffix=STRICT-REGEX         %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomFunctions: '^name_match$,replacement,is matched on function name only;^::prefix_match$,,is a full qualname match;^::S::member_match_1$,,is matched on a C++ class member'}}"

void name_match();
void prefix_match();

struct S {
  static void member_match_1() {}
  void member_match_2() {}
};

void member_match_1() {}
void member_match_unmatched() {}

namespace regex_test {
void name_match();
void prefix_match();
}

void name_match_regex();
void prefix_match_regex();

void f1() {
  name_match();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'name_match' is a qualname match; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'name_match' is matched on function name only; 'replacement' should be used instead
  prefix_match();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'prefix_match' is matched on qualname prefix; it should not be used
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'prefix_match' is a full qualname match; it should not be used

  ::name_match();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'name_match' is a qualname match; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'name_match' is matched on function name only; 'replacement' should be used instead
  regex_test::name_match();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'name_match' is a qualname match; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'name_match' is matched on function name only; 'replacement' should be used instead
  name_match_regex();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'name_match_regex' is a qualname match; 'replacement' should be used instead
  // no-warning STRICT-REGEX

  ::prefix_match();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'prefix_match' is matched on qualname prefix; it should not be used
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'prefix_match' is a full qualname match; it should not be used
  regex_test::prefix_match();
  // no-warning NON-STRICT-REGEX
  // no-warning STRICT-REGEX
  prefix_match_regex();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'prefix_match_regex' is matched on qualname prefix; it should not be used
  // no-warning STRICT-REGEX
}

void f2() {
  S s;

  S::member_match_1();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'member_match_1' is matched on a C++ class member; it should not be used
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'member_match_1' is matched on a C++ class member; it should not be used

  s.member_match_1();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:5: warning: function 'member_match_1' is matched on a C++ class member; it should not be used
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:5: warning: function 'member_match_1' is matched on a C++ class member; it should not be used

  s.member_match_2();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:5: warning: function 'member_match_2' is matched on a C++ class member; it should not be used
  // no-warning STRICT-REGEX

  member_match_1();
  // no-warning

  member_match_unmatched();
  // no-warning
}
