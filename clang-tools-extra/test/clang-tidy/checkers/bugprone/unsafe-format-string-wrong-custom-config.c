// RUN: %check_clang_tidy %s bugprone-unsafe-format-string %t --\
// RUN:  -config="{CheckOptions: {bugprone-unsafe-format-string.CustomPrintfFunctions: 'mysprintf, 0; mylogger, 1', bugprone-unsafe-format-string.CustomScanfFunctions: 'myscanf, 0;'  }}"\
// RUN: -- -I %S/../Inputs/Headers/std

#include "system-header-simulator.h"

extern int myscanf( const char* format, ... );
extern int mysprintf( char* buffer, const char* format, ... );
extern int mylogger( char* buffer, const char* format, ... );

void test_sprintf() {
  char buffer[100];
  const char* input = "user input";

  /* unsafe %s without field width */
  mysprintf(buffer, "%s", input);
  // no warning as mysprintf config is wrong

  mylogger(buffer, "%s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  mysprintf(buffer, "%.99s", input);
  /*no warning*/
}


void test_scanf() {
  char buffer[100];

  /* unsafe %s without field width */
  myscanf("%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  myscanf("%99s", buffer);
  /* no-warning */
}
