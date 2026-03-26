// RUN: %clang_cc1 -fsyntax-only -Wenum-compare-typo -verify %s 
// RUN: %clang_cc1 -fsyntax-only -Wenum-compare-typo -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s


enum PossibleTypoLeft { 
  Val1 = 1 << 0,
  // expected-warning@+3 {{comparison operator '<' is potentially a typo for a shift operator '<<'}} 
  // expected-note@+2 {{use '<<' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:12-[[@LINE+1]]:13}:"<<"
  Bad1 = 1 < 2, 
  // expected-warning@+3 {{comparison operator '>' is potentially a typo for a shift operator '>>'}} 
  // expected-note@+2 {{use '>>' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:12-[[@LINE+1]]:13}:">>"
  Bad2 = 1 > 3,
  // expected-warning@+3 {{comparison operator '>' is potentially a typo for a shift operator '>>'}} 
  // expected-note@+2 {{use '>>' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:13-[[@LINE+1]]:14}:">>"
  Bad3 = (1 > 3) 
};

enum PossibleTypoRight { 
  Val2 = 1 >> 0,
  // expected-warning@+3 {{comparison operator '<' is potentially a typo for a shift operator '<<'}} 
  // expected-note@+2 {{use '<<' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:12-[[@LINE+1]]:13}:"<<"
  Bad4 = 1 < 2, 
  // expected-warning@+3 {{comparison operator '>' is potentially a typo for a shift operator '>>'}} 
  // expected-note@+2 {{use '>>' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:12-[[@LINE+1]]:13}:">>"
  Bad5 = 1 > 3,
  // expected-warning@+3 {{comparison operator '<' is potentially a typo for a shift operator '<<'}} 
  // expected-note@+2 {{use '<<' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:13-[[@LINE+1]]:14}:"<<"
  Bad6 = (1 < 3) 
};

// Case 3: Context provided by other bitwise operators (&, |)
// Even though there are no shifts, the presence of '|' implies flags.
enum PossibleTypoBitwiseOr {
  FlagA = 0x1,
  FlagB = 0x2,
  FlagCombo = FlagA | FlagB,
  // expected-warning@+3 {{comparison operator '<' is potentially a typo for a shift operator '<<'}} 
  // expected-note@+2 {{use '<<' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:17-[[@LINE+1]]:18}:"<<"
  FlagTypo1 = 1 < FlagCombo,
  // expected-warning@+3 {{comparison operator '>' is potentially a typo for a shift operator '>>'}} 
  // expected-note@+2 {{use '>>' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:17-[[@LINE+1]]:18}:">>"
  FlagTypo2 = 1 > FlagCombo
};

enum PossibleTypoBitwiseAnd {
  FlagAnd = FlagA & FlagB,
  // expected-warning@+3 {{comparison operator '<' is potentially a typo for a shift operator '<<'}} 
  // expected-note@+2 {{use '<<' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:17-[[@LINE+1]]:18}:"<<"
  FlagTypo3 = 1 < FlagAnd,
  // expected-warning@+3 {{comparison operator '>' is potentially a typo for a shift operator '>>'}} 
  // expected-note@+2 {{use '>>' to perform a bitwise shift}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE+1]]:17-[[@LINE+1]]:18}:">>"
  FlagTypo4 = 1 > FlagAnd
};

enum NoWarningOnDirectInit {
  A = 0,
  B = 1,
  Ok1 = 1 < 2, // No warning expected
  Ok2 = 1 > 2 // No warning expected
};

enum NoWarningOnArith {
  D = 0 + 1,
  E = D * 10,
  F = E - D,
  G = F / D,
  Ok3 = 1 < E, // No warning expected
  Ok4 = 1 > E // No warning expected
};

enum NoWarningOnExplicitCast {
  Bit1 = 1 << 0,
  Ok5 = (int)(1 < 2) // No warning expected
};

enum NoWarningOnNoneBitShift {
  Bit2 = 1 << 0,
  Ok6 = (3 < 2) // No warning expected
};

// Ensure the diagnostic group works
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wenum-compare-typo"
enum IGNORED {
  Ok7 = 1 << 1,
  Ignored3 = 1 < 10 // No warning
};
#pragma clang diagnostic pop
