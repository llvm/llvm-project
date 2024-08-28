// This test fails on "x86_64-sie" buildbot and "x86_64-scei-ps4" target.
// According to @dyung, something related to the kind of standard library
// availability is causing the failure. Even though we explicitly define
// the relevant macros the check is hunting for in the invocation, the real
// parsing and preprocessor state will not have that case.
// UNSUPPORTED: target={{.*-(ps4|ps5)}}
//
// RUN: %check_clang_tidy -check-suffix=WITHOUT-ANNEX-K         %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomNormalFunctions: '^non_annex_k_only$,replacement;^both$,replacement,is a normal function', bugprone-unsafe-functions.CustomAnnexKFunctions: '^annex_k_only$,replacement;^both$,replacement,is an Annex K function'}}"\
// RUN:   -- -U__STDC_LIB_EXT1__   -U__STDC_WANT_LIB_EXT1__
// RUN: %check_clang_tidy -check-suffix=WITH-ANNEX-K            %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomNormalFunctions: '^non_annex_k_only$,replacement;^both$,replacement,is a normal function', bugprone-unsafe-functions.CustomAnnexKFunctions: '^annex_k_only$,replacement;^both$,replacement,is an Annex K function'}}"\
// RUN:   -- -D__STDC_LIB_EXT1__=1 -D__STDC_WANT_LIB_EXT1__=1
// RUN: %check_clang_tidy -check-suffix=WITH-ONLY-ANNEX-K       %s bugprone-unsafe-functions %t --\
// RUN:   -config="{CheckOptions: {bugprone-unsafe-functions.CustomAnnexKFunctions: '^annex_k_only$,replacement;^both$,replacement,is an Annex K function'}}"\
// RUN:   -- -D__STDC_LIB_EXT1__=1 -D__STDC_WANT_LIB_EXT1__=1

void non_annex_k_only(void);
void annex_k_only(void);
void both(void);

void f1() {
  non_annex_k_only();
  // CHECK-MESSAGES-WITHOUT-ANNEX-K: :[[@LINE-1]]:3: warning: function 'non_annex_k_only' is marked as unsafe; 'replacement' should be used instead
  // CHECK-MESSAGES-WITH-ANNEX-K: :[[@LINE-2]]:3: warning: function 'non_annex_k_only' is marked as unsafe; 'replacement' should be used instead
  // no-warning WITH-ONLY-ANNEX-K
  annex_k_only();
  // no-warning WITHOUT-ANNEX-K
  // CHECK-MESSAGES-WITH-ANNEX-K: :[[@LINE-2]]:3: warning: function 'annex_k_only' is marked as unsafe; 'replacement' should be used instead
  // CHECK-MESSAGES-WITH-ONLY-ANNEX-K: :[[@LINE-3]]:3: warning: function 'annex_k_only' is marked as unsafe; 'replacement' should be used instead
  both();
  // CHECK-MESSAGES-WITH-ANNEX-K: :[[@LINE-1]]:3: warning: function 'both' is an Annex K function; 'replacement' should be used instead
  // CHECK-MESSAGES-WITHOUT-ANNEX-K: :[[@LINE-2]]:3: warning: function 'both' is a normal function; 'replacement' should be used instead
  // CHECK-MESSAGES-WITH-ONLY-ANNEX-K: :[[@LINE-3]]:3: warning: function 'both' is an Annex K function; 'replacement' should be used instead
}
