// RUN: %check_clang_tidy %s readability-suspicious-call-argument %t \
// RUN: -config="{CheckOptions: {readability-suspicious-call-argument.Abbreviations: 'crash='}}" -- -std=c++11-or-later
// RUN: %check_clang_tidy %s readability-suspicious-call-argument %t -check-suffix=MINLEN \
// RUN: -config='{CheckOptions: {readability-suspicious-call-argument.MinimumIdentifierNameLength: 10}}' -- -std=c++11-or-later

void f() {}
// CHECK-MESSAGES: warning: Invalid abbreviation configuration 'crash=', ignoring.

void takeTwoParams(int frobble1, int frobble2);

void testMinimumIdentifierNameLength() {
  int frobble2 = 1, frobble1 = 2;
  takeTwoParams(frobble2, frobble1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: 1st argument 'frobble2' (passed to 'frobble1') looks like it might be swapped with the 2nd, 'frobble1' (passed to 'frobble2')
  // No warning for MINLEN
}
