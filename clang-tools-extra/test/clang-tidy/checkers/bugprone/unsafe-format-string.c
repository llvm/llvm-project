// RUN: %check_clang_tidy %s bugprone-unsafe-format-string %t -- -- -I %S/../Inputs/Headers/std

#include "system-header-simulator.h"

void test_sprintf() {
  char buffer[100];
  const char* input = "user input";

  /* unsafe %s without field width */
  sprintf(buffer, "%s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* field width doesn't prevent overflow in sprintf */
  sprintf(buffer, "%99s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* dynamic field width doesn't prevent overflow */
  sprintf(buffer, "%*s", 10, input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /*precision limits string length */
  sprintf(buffer, "%.99s", input);

  /*precision with field width */
  sprintf(buffer, "%1.99s", input);

  /*dynamic precision */
  sprintf(buffer, "%.*s", 99, input);

  /*field width with dynamic precision */
  sprintf(buffer, "%1.*s", 99, input);

  /*dynamic field width with fixed precision */
  sprintf(buffer, "%*.99s", 10, input);

  /*dynamic field width and precision */
  sprintf(buffer, "%*.*s", 10, 99, input);

  /*limit is taken as 0*/
  sprintf(buffer, "%.s", 99, input);

  /*other format specifiers are safe */
  sprintf(buffer, "%d %f", 42, 3.14);

  //Tests taken from warn-format-overflow-truncation.c
  sprintf(buffer, "hell\0 boy");
  sprintf(buffer, "hello b\0y");
  sprintf(buffer, "hello");
  sprintf(buffer, "hello!");
  sprintf(buffer, "1234%%");
  sprintf(buffer, "12345%%");
  sprintf(buffer, "1234%c", '9');
  sprintf(buffer, "12345%c", '9');
  sprintf(buffer, "1234%d", 9);
  sprintf(buffer, "12345%d", 9);
  sprintf(buffer, "1234%lld", 9ll);
  sprintf(buffer, "12345%lld", 9ll);
  sprintf(buffer, "12%#x", 9);
  sprintf(buffer, "123%#x", 9);
  sprintf(buffer, "12%p", (void *)9);
  sprintf(buffer, "123%p", (void *)9);
  sprintf(buffer, "123%+d", 9);
  sprintf(buffer, "1234%+d", 9);
  sprintf(buffer, "123% i", 9);
  sprintf(buffer, "1234% i", 9);
  sprintf(buffer, "%5d", 9);
  sprintf(buffer, "1%5d", 9);
  sprintf(buffer, "%.3f", 9.f);
  sprintf(buffer, "5%.3f", 9.f);
  sprintf(buffer, "%+.2f", 9.f);
  sprintf(buffer, "%+.3f", 9.f);
  sprintf(buffer, "%.0e", 9.f);
  sprintf(buffer, "5%.1e", 9.f);
  sprintf(buffer, "%5.1f", 9.f);
  sprintf(buffer, "%+5.1f", 9.f);
  sprintf(buffer, "% 5.1f", 9.f);
  sprintf(buffer, "%   5.1f", 9.f);
  sprintf(buffer, "%+6.1f", 9.f);
  sprintf(buffer, "% 6.1f", 9.f);
  sprintf(buffer, "%   6.1f", 9.f);

}

void test_vsprintf() {
  char buffer[100];
  va_list args;

  /* unsafe %s without field width */
  vsprintf(buffer, "%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* field width doesn't prevent overflow in vsprintf */
  vsprintf(buffer, "%99s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* precision limits string length */
  vsprintf(buffer, "%.99s", args);

}

void test_vsnprintf(int count, ...) {
  va_list args;
  va_start(args, count);
  char buffer[100];

  /*vsnprintf is safe */
  vsnprintf(buffer, sizeof(buffer), "%99s", args);

  va_end(args);
}

void test_scanf() {
  char buffer[100];
  char buffer2[100];
  int i;

  /* unsafe %s without field width */
  scanf("%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  scanf("%99s %s", buffer, buffer2);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  scanf("%*s %s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  scanf("%%%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  scanf("%99s", buffer);

  /*safe %*s does not write into buffer */
  scanf("%*s %99s", buffer);

  scanf("%%%99s", buffer);

  scanf("%ds", &i);

}

void test_fscanf() {
  char buffer[100];
  FILE* file = 0;

  /* unsafe %s without field width */
  fscanf(file, "%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  fscanf(file, "%99s", buffer);
}

void test_sscanf(char *source) {
  char buffer[100];

  /* unsafe %s without field width */
  sscanf(source, "%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  sscanf(source, "%99s", buffer);
}

void test_vfscanf() {
  FILE* file = 0;
  va_list args;

  /* unsafe %s without field width */
  vfscanf(file, "%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  vfscanf(file, "%99s", args);
}

void test_vsscanf(char * source) {
  va_list args;

  /* unsafe %s without field width */
  vsscanf(source, "%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  vsscanf(source, "%99s", args);
}

void test_vscanf() {
  va_list args;

  /* unsafe %s without field width */
  vscanf("%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  vscanf("%99s", args);
}

void test_wscanf() {
  wchar_t buffer[100];

  /* unsafe %s without field width */
  wscanf(L"%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  wscanf(L"%99s", buffer);
}

void test_fwscanf() {
  wchar_t buffer[100];
  FILE* file = 0;

  /* unsafe %s without field width */
  fwscanf(file, L"%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  fwscanf(file, L"%99s", buffer);
}

void test_swscanf(wchar_t *source) {
  wchar_t buffer[100];

  /* unsafe %s without field width */
  swscanf(source, L"%s", buffer);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  swscanf(source, L"%99s", buffer);
}

void test_vwscanf() {
  va_list args;

  /* unsafe %s without field width */
  vwscanf(L"%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  vwscanf(L"%99s", args);
}

void test_vfwscanf() {
  FILE* file = 0;
  va_list args;

  /* unsafe %s without field width */
  vfwscanf(file, L"%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  vfwscanf(file, L"%99s", args);
}

void test_vswscanf() {
  const wchar_t* source = L"input";
  va_list args;

  /* unsafe %s without field width */
  vswscanf(source, L"%s", args);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without field width may cause buffer overflow; consider using '%Ns' where N limits input length [bugprone-unsafe-format-string]

  /*safe %s with field width */
  vswscanf(source, L"%99s", args);
}

void test_safe_alternatives() {
  char buffer[100];
  const char* input = "user input";

  /*snprintf is inherently safe */
  snprintf(buffer, sizeof(buffer), "%s", input);

  /*printf family doesn't write to buffers */
  printf("%s", input);

  /*fprintf doesn't write to user buffers */
  fprintf(stderr, "%s", input);
}
