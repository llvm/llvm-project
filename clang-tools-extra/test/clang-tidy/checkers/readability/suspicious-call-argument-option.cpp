// RUN: %check_clang_tidy %s readability-suspicious-call-argument %t \
// RUN: -config="{CheckOptions: {readability-suspicious-call-argument.Abbreviations: 'crash='}}" -- -std=c++11-or-later

void f() {}
// CHECK-MESSAGES: warning: Invalid abbreviation configuration 'crash=', ignoring.

// TODO: Add testcases for other options
