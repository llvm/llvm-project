// Same as invalid-analyzer-config-value.c but without -analyzer-config
// in the file name, so that argument string pattern matching
// didn't accidentally match it.

// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config notes-as-events=yesplease \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-BOOL-INPUT

// CHECK-BOOL-INPUT: (frontend): invalid input for analyzer-config option
// CHECK-BOOL-INPUT-SAME:        'notes-as-events', that expects a boolean value

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config notes-as-events=yesplease


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config max-inlinable-size=400km/h \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-UINT-INPUT

// CHECK-UINT-INPUT: (frontend): invalid input for analyzer-config option
// CHECK-UINT-INPUT-SAME:        'max-inlinable-size', that expects an unsigned
// CHECK-UINT-INPUT-SAME:        value

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config max-inlinable-size=400km/h


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config ctu-dir=0123012301230123 \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-FILENAME-INPUT

// CHECK-FILENAME-INPUT: (frontend): invalid input for analyzer-config option
// CHECK-FILENAME-INPUT-SAME:        'ctu-dir', that expects a filename
// CHECK-FILENAME-INPUT-SAME:        value

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config ctu-dir=0123012301230123


// RUN: not %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config no-false-positives=true \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-UNKNOWN-CFG

// CHECK-UNKNOWN-CFG: (frontend): unknown analyzer-config 'no-false-positives'

// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config-compatibility-mode=true \
// RUN:   -analyzer-config no-false-positives=true


// Test the driver properly using "analyzer-config-compatibility-mode=true",
// no longer causing an error on input error.
// RUN: %clang --analyze %s

// RUN: not %clang --analyze %s \
// RUN:   -Xclang -analyzer-config -Xclang no-false-positives=true \
// RUN:   -Xclang -analyzer-config-compatibility-mode=false \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-NO-COMPAT

// CHECK-NO-COMPAT: error: unknown analyzer-config 'no-false-positives'

// Test the driver properly using "analyzer-config-compatibility-mode=true",
// even if -analyze isn't specified.
// RUN: %clang -fsyntax-only -Xclang -analyzer-config\
// RUN:                      -Xclang remember=TheVasa %s
// RUN: %clang -fsyntax-only -Xanalyzer -analyzer-config\
// RUN:                      -Xanalyzer remember=TheVasa %s
// RUN: %clang --analyze -Xanalyzer -analyzer-config\
// RUN:                  -Xanalyzer remember=TheVasa %s

// expected-no-diagnostics

int main(void) {}
