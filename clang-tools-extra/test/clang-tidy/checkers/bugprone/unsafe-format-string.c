// RUN: %check_clang_tidy %s bugprone-unsafe-format-string %t -- -- -isystem %S/Inputs/unsafe-format-string

#include <system-header-simulator.h>

void test_sprintf() {
  char buffer[100];
  const char* input = "user input";

  /* Positive: unsafe %s without field width */
  sprintf(buffer, "%s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* Positive: field width doesn't prevent overflow in sprintf */
  sprintf(buffer, "%99s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* Positive: dynamic field width doesn't prevent overflow */
  sprintf(buffer, "%*s", 10, input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /*Negative: precision limits string length */
  sprintf(buffer, "%.99s", input);
  /* no-warning */

  /*Negative: precision with field width */
  sprintf(buffer, "%1.99s", input);
  /* no-warning */

  /*Negative: dynamic precision */
  sprintf(buffer, "%.*s", 99, input);
  /* no-warning */

  /*Negative: field width with dynamic precision */
  sprintf(buffer, "%1.*s", 99, input);
  /* no-warning */

  /*Negative: dynamic field width with fixed precision */
  sprintf(buffer, "%*.99s", 10, input);
  /* no-warning */

  /*Negative: dynamic field width and precision */
  sprintf(buffer, "%*.*s", 10, 99, input);
  /* no-warning */

  /*Negative: other format specifiers are safe */
  sprintf(buffer, "%d %f", 42, 3.14);
  /* no-warning */
}

void test_vsprintf() {
  char buffer[100];
  va_list args;

  /* Positive: unsafe %s without field width */
  vsprintf(buffer, "%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* Positive: field width doesn't prevent overflow in vsprintf */
  vsprintf(buffer, "%99s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* Positive: precision limits string length */
  vsprintf(buffer, "%.99s", args);
  /* no-warning */
}

void test_vsnprintf(int count, ...) {
  va_list args;
  va_start(args, count);
  char buffer[100];

  /*Negative: vsnprintf is safe */
  vsnprintf(buffer, sizeof(buffer), "%99s", args);
  /* no-warning */

  va_end(args);
}

void test_scanf() {
  char buffer[100];

  /* Positive: unsafe %s without field width */
  scanf("%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  scanf("%99s", buffer);
  /* no-warning */
}

void test_fscanf() {
  char buffer[100];
  FILE* file = 0;

  /* Positive: unsafe %s without field width */
  fscanf(file, "%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  fscanf(file, "%99s", buffer);
  /* no-warning */
}

void test_sscanf(char *source) {
  char buffer[100];

  /* Positive: unsafe %s without field width */
  sscanf(source, "%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  sscanf(source, "%99s", buffer);
  /* no-warning */
}

void test_vfscanf() {
  FILE* file = 0;
  va_list args;

  /* Positive: unsafe %s without field width */
  vfscanf(file, "%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  vfscanf(file, "%99s", args);
  /* no-warning */
}

void test_vsscanf(char * source) {
  va_list args;

  /* Positive: unsafe %s without field width */
  vsscanf(source, "%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  vsscanf(source, "%99s", args);
  /* no-warning */
}

void test_vscanf() {
  va_list args;

  /* Positive: unsafe %s without field width */
  vscanf("%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  vscanf("%99s", args);
  /* no-warning */
}

void test_wscanf() {
  wchar_t buffer[100];

  /* Positive: unsafe %s without field width */
  wscanf(L"%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  wscanf(L"%99s", buffer);
  /* no-warning */
}

void test_fwscanf() {
  wchar_t buffer[100];
  FILE* file = 0;

  /* Positive: unsafe %s without field width */
  fwscanf(file, L"%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  fwscanf(file, L"%99s", buffer);
  /* no-warning */
}

void test_swscanf(wchar_t *source) {
  wchar_t buffer[100];

  /* Positive: unsafe %s without field width */
  swscanf(source, L"%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  swscanf(source, L"%99s", buffer);
  /* no-warning */
}

void test_vwscanf() {
  va_list args;

  /* Positive: unsafe %s without field width */
  vwscanf(L"%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  vwscanf(L"%99s", args);
  /* no-warning */
}

void test_vfwscanf() {
  FILE* file = 0;
  va_list args;

  /* Positive: unsafe %s without field width */
  vfwscanf(file, L"%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  vfwscanf(file, L"%99s", args);
  /* no-warning */
}

void test_vswscanf() {
  const wchar_t* source = L"input";
  va_list args;

  /* Positive: unsafe %s without field width */
  vswscanf(source, L"%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*Negative: safe %s with field width */
  vswscanf(source, L"%99s", args);
  /* no-warning */
}

void test_safe_alternatives() {
  char buffer[100];
  const char* input = "user input";

  /*Negative: snprintf is inherently safe */
  snprintf(buffer, sizeof(buffer), "%s", input);
  /* no-warning */

  /*Negative: printf family doesn't write to buffers */
  printf("%s", input);
  /* no-warning */

  /*Negative: fprintf doesn't write to user buffers */
  fprintf(stderr, "%s", input);
  /* no-warning */
}
