// RUN: %check_clang_tidy %s bugprone-unsafe-format-string %t --\
// RUN:  -config="{CheckOptions: {bugprone-unsafe-format-string.CustomScanfFunctions: 'myscanf, wrong;'  }}"

// CHECK-MESSAGES: warning: invalid configuration value for option 'CustomScanfFunctions'