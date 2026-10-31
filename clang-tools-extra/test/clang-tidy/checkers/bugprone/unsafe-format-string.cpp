// RUN: %check_clang_tidy %s bugprone-unsafe-format-string %t -- --  -I %S/../Inputs/Headers/std

#include "system-header-simulator.h"

class TestClass{
  int sprintf( char* buffer, const char* format, ... );
  void test_sprintf() {
    char buffer[100];
    const char* input = "user input";
    /* no warning for calling member functions */
    sprintf(buffer, "%s", input);
  }
};

void test_sprintf() {
  char buffer[100];
  const char* input = "user input";

  /* unsafe %s without field width */
  std::sprintf(buffer, "%s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* field width doesn't prevent overflow in sprintf */
  std::sprintf(buffer, "%99s", input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /* dynamic field width doesn't prevent overflow */
  std::sprintf(buffer, "%*s", 10, input);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: format specifier '%s' without precision may cause buffer overflow; consider using '%.Ns' where N limits output length [bugprone-unsafe-format-string]

  /*precision limits string length */
  std::sprintf(buffer, "%.99s", input);

  /*precision with field width */
  std::sprintf(buffer, "%1.99s", input);

  /*dynamic precision */
  std::sprintf(buffer, "%.*s", 99, input);

  /*field width with dynamic precision */
  std::sprintf(buffer, "%1.*s", 99, input);

  /*dynamic field width with fixed precision */
  std::sprintf(buffer, "%*.99s", 10, input);

  /*dynamic field width and precision */
  std::sprintf(buffer, "%*.*s", 10, 99, input);


  /*other format specifiers are safe */
  std::sprintf(buffer, "%d %f", 42, 3.14);
}
