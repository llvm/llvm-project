// This test fails on "x86_64-sie" buildbot and "x86_64-scei-ps4" target.
// According to @dyung, something related to the kind of standard library
// availability is causing the failure. Even though we explicitly define
// the relevant macros the check is hunting for in the invocation, the real
// parsing and preprocessor state will not have that case.
// UNSUPPORTED: target={{.*-(ps4|ps5)}}
//
// RUN: %check_clang_tidy -check-suffix=NON-STRICT-REGEX         %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomNormalFunctions: '::non_annex_k_only,replacement;::both,replacement,is a normal function'}}"\
// RUN:   -- -U__STDC_LIB_EXT1__   -U__STDC_WANT_LIB_EXT1__
// RUN: %check_clang_tidy -check-suffix=STRICT-REGEX         %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomNormalFunctions: '^non_annex_k_only$,replacement,is matched on function name only;^::both,replacement,is matched on qualname prefix'}}"\
// RUN:   -- -U__STDC_LIB_EXT1__   -U__STDC_WANT_LIB_EXT1__
// RUN: %check_clang_tidy -check-suffix=WITH-ANNEX-K       %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomAnnexKFunctions: '::annex_k_only,replacement;::both,replacement,is an Annex K function'}}"\
// RUN:   -- -D__STDC_LIB_EXT1__=1 -D__STDC_WANT_LIB_EXT1__=1

// TODO: How is "Annex K" available in C++ mode?
// no-warning WITH-ANNEX-K

void non_annex_k_only();
void annex_k_only();
void both();

namespace regex_test {
void non_annex_k_only();
void both();
}

void non_annex_k_only_regex();
void both_regex();

void f1() {
  non_annex_k_only();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'non_annex_k_only' is marked as unsafe; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'non_annex_k_only' is matched on function name only; 'replacement' should be used instead
  annex_k_only();
  // no-warning NON-STRICT-REGEX
  // no-warning STRICT-REGEX
  both();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'both' is a normal function; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'both' is matched on qualname prefix; 'replacement' should be used instead

  ::non_annex_k_only();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'non_annex_k_only' is marked as unsafe; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'non_annex_k_only' is matched on function name only; 'replacement' should be used instead
  regex_test::non_annex_k_only();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'non_annex_k_only' is marked as unsafe; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'non_annex_k_only' is matched on function name only; 'replacement' should be used instead
  non_annex_k_only_regex();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'non_annex_k_only_regex' is marked as unsafe; 'replacement' should be used instead
  // no-warning STRICT-REGEX

  ::both();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'both' is a normal function; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'both' is matched on qualname prefix; 'replacement' should be used instead
  regex_test::both();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'both' is a normal function; 'replacement' should be used instead
  // no-warning STRICT-REGEX
  both_regex();
  // CHECK-MESSAGES-NON-STRICT-REGEX: :[[@LINE-1]]:3: warning: function 'both_regex' is a normal function; 'replacement' should be used instead
  // CHECK-MESSAGES-STRICT-REGEX: :[[@LINE-2]]:3: warning: function 'both_regex' is matched on qualname prefix; 'replacement' should be used instead
}
