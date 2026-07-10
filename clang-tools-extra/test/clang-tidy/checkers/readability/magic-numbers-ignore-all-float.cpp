// RUN: %check_clang_tidy %s readability-magic-numbers %t -check-suffix=IGNFP \
// RUN: -config='{CheckOptions: {readability-magic-numbers.IgnoreAllFloatingPointValues: true}}' --

int BadInt = 5;
// CHECK-MESSAGES-IGNFP: :[[@LINE-1]]:14: warning: 5 is a magic number; consider replacing it with a named constant [readability-magic-numbers]

float IgnoredFloat = 3.14f;
// CHECK-MESSAGES-IGNFP-NOT: 3.14f is a magic number
