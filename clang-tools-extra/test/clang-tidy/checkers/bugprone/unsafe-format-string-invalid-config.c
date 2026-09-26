// RUN: %check_clang_tidy %s bugprone-unsafe-format-string %t --\
// RUN:  -config="{CheckOptions: {bugprone-unsafe-format-string.CustomPrintfFunctions: 'mysprintf, wrong-value; mylogger, 1', bugprone-unsafe-format-string.CustomScanfFunctions: 'myscanf, 0;'  }}"\
// RUN: -- -isystem %S/Inputs/unsafe-format-string

// CHECK-MESSAGES: warning: invalid configuration value for option