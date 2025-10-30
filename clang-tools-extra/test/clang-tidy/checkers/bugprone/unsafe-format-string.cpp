// RUN: %check_clang_tidy %s bugprone-unsafe-format-string %t -- -- -isystem %S/Inputs/unsafe-format-string

#include <system-header-simulator.h>

class TestClass{
  int sprintf( char* buffer, const char* format, ... );
  void test_sprintf() {
    char buffer[100];
    const char* input = "user input";
    /* Negative: no warning for calling member functions */
    sprintf(buffer, "%s", input);
  }
};

void test_sprintf() {
  char buffer[100];
  const char* input = "user input";

  /* Positive: unsafe %s without field width */
  std::sprintf(buffer, "%s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* Positive: field width doesn't prevent overflow in sprintf */
  std::sprintf(buffer, "%99s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* Positive: dynamic field width doesn't prevent overflow */
  std::sprintf(buffer, "%*s", 10, input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /*Negative: precision limits string length */
  std::sprintf(buffer, "%.99s", input);
  /* no-warning */

  /*Negative: precision with field width */
  std::sprintf(buffer, "%1.99s", input);
  /* no-warning */

  /*Negative: dynamic precision */
  std::sprintf(buffer, "%.*s", 99, input);
  /* no-warning */

  /*Negative: field width with dynamic precision */
  std::sprintf(buffer, "%1.*s", 99, input);
  /* no-warning */

  /*Negative: dynamic field width with fixed precision */
  std::sprintf(buffer, "%*.99s", 10, input);
  /* no-warning */

  /*Negative: dynamic field width and precision */
  std::sprintf(buffer, "%*.*s", 10, 99, input);
  /* no-warning */

  /*Negative: other format specifiers are safe */
  std::sprintf(buffer, "%d %f", 42, 3.14);
  /* no-warning */
}
