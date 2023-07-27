// RUN: %check_clang_tidy -check-suffixes=,STRICT \
// RUN:   -std=c++23 %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: [{key: StrictMode, value: true}]}" \
// RUN:   -- -isystem %clang_tidy_headers -fexceptions
// RUN: %check_clang_tidy -check-suffixes=,NOTSTRICT \
// RUN:   -std=c++23 %s modernize-use-std-print %t -- \
// RUN:   -config="{CheckOptions: [{key: StrictMode, value: false}]}" \
// RUN:   -- -isystem %clang_tidy_headers -fexceptions
#include <cstddef>
#include <cstdint>
#include <cstdio>
// CHECK-FIXES: #include <print>
#include <inttypes.h>
#include <string.h>
#include <string>

void printf_simple() {
  printf("Hello");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Hello");
}

void printf_newline() {
  printf("Hello\n");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello");

  printf("Split" "\n");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Split");

  printf("Double\n\n");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Double\n");
}

void printf_deceptive_newline() {
  printf("Hello\\n");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Hello\\n");

  printf("Hello\x0a");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello");
}

void printf_crlf_newline() {
  printf("Hello\r\n");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Hello\r\n");

  printf("Hello\r\\n");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Hello\r\\n");
}

// std::print returns nothing, so any callers that use the return
// value cannot be automatically translated.
int printf_uses_return_value(int choice) {
  const int i = printf("Return value assigned to variable %d\n", 42);

  extern void accepts_int(int);
  accepts_int(printf("Return value passed to function %d\n", 42));

  if (choice == 0)
    printf("if body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("if body {}", i);
  else if (choice == 1)
    printf("else if body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("else if body {}", i);
  else
    printf("else body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("else body {}", i);

  if (printf("Return value used as boolean in if statement"))
    if (printf("Return value used in expression if statement") == 44)
      if (const int j = printf("Return value used in assignment in if statement"))
        if (const int k = printf("Return value used with initializer in if statement"); k == 44)
          ;

  int d = 0;
  while (printf("%d", d) < 2)
    ++d;

  while (true)
    printf("while body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("while body {}", i);

  do
    printf("do body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("do body {}", i);
  while (true);

  for (;;)
    printf("for body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("for body {}", i);

  for (printf("for init statement %d\n", i);;)
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("for init statement {}", i);
    ;;

  for (int j = printf("for init statement %d\n", i);;)
    ;;

  for (; printf("for condition %d\n", i);)
    ;;

  for (;; printf("for expression %d\n", i))
    // CHECK-MESSAGES: [[@LINE-1]]:11: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("for expression {}", i)
    ;;

  for (auto C : "foo")
    printf("ranged-for body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("ranged-for body {}", i);

  switch (1) {
  case 1:
    printf("switch case body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("switch case body {}", i);
    break;
  default:
    printf("switch default body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("switch default body {}", i);
    break;
  }

  try {
    printf("try body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("try body {}", i);
  } catch (int) {
    printf("catch body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("catch body {}", i);
  }

  (printf("Parenthesised expression %d\n", i));
  // CHECK-MESSAGES: [[@LINE-1]]:4: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: (std::println("Parenthesised expression {}", i));

  // Ideally we would convert these two, but the current check doesn't cope with
  // that.
  (void)printf("cast to void %d\n", i);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:9: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println("cast to void {}", i);

  static_cast<void>(printf("static_cast to void %d\n", i));
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:9: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println("static cast to void {}", i);

  const int x = ({ printf("GCC statement expression using return value immediately %d\n", i); });
  const int y = ({ const int y = printf("GCC statement expression using return value immediately %d\n", i); y; });

  // Ideally we would convert this one, but the current check doesn't cope with
  // that.
  ({ printf("GCC statement expression with unused result %d\n", i); });
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:6: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println("GCC statement expression with unused result {}", i);

  return printf("Return value used in return\n");
}

int fprintf_uses_return_value(int choice) {
  const int i = fprintf(stderr, "Return value assigned to variable %d\n", 42);

  extern void accepts_int(int);
  accepts_int(fprintf(stderr, "Return value passed to function %d\n", 42));

  if (choice == 0)
    fprintf(stderr, "if body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "if body {}", i);
  else if (choice == 1)
    fprintf(stderr, "else if body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "else if body {}", i);
  else
    fprintf(stderr, "else body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "else body {}", i);

  if (fprintf(stderr, "Return value used as boolean in if statement"))
    if (fprintf(stderr, "Return value used in expression if statement") == 44)
      if (const int j = fprintf(stderr, "Return value used in assignment in if statement"))
        if (const int k = fprintf(stderr, "Return value used with initializer in if statement"); k == 44)
          ;

  int d = 0;
  while (fprintf(stderr, "%d", d) < 2)
    ++d;

  while (true)
    fprintf(stderr, "while body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "while body {}", i);

  do
    fprintf(stderr, "do body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "do body {}", i);
  while (true);

  for (;;)
    fprintf(stderr, "for body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "for body {}", i);

  for (fprintf(stderr, "for init statement %d\n", i);;)
    // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "for init statement {}", i);
    ;;

  for (int j = fprintf(stderr, "for init statement %d\n", i);;)
    ;;

  for (; fprintf(stderr, "for condition %d\n", i);)
    ;;

  for (;; fprintf(stderr, "for expression %d\n", i))
    // CHECK-MESSAGES: [[@LINE-1]]:11: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "for expression {}", i)
    ;;

  for (auto C : "foo")
    fprintf(stderr, "ranged-for body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "ranged-for body {}", i);

  switch (1) {
  case 1:
    fprintf(stderr, "switch case body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "switch case body {}", i);
    break;
  default:
    fprintf(stderr, "switch default body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "switch default body {}", i);
    break;
  }

  try {
    fprintf(stderr, "try body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "try body {}", i);
  } catch (int) {
    fprintf(stderr, "catch body %d\n", i);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
    // CHECK-FIXES: std::println(stderr, "catch body {}", i);
  }


  (printf("Parenthesised expression %d\n", i));
  // CHECK-MESSAGES: [[@LINE-1]]:4: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: (std::println("Parenthesised expression {}", i));

  // Ideally we would convert these two, but the current check doesn't cope with
  // that.
  (void)fprintf(stderr, "cast to void %d\n", i);
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:9: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println(stderr, "cast to void {}", i);

  static_cast<void>(fprintf(stderr, "static_cast to void %d\n", i));
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:9: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println(stderr, "static cast to void {}", i);

  const int x = ({ fprintf(stderr, "GCC statement expression using return value immediately %d\n", i); });
  const int y = ({ const int y = fprintf(stderr, "GCC statement expression using return value immediately %d\n", i); y; });

  // Ideally we would convert this one, but the current check doesn't cope with
  // that.
  ({ fprintf(stderr, "GCC statement expression with unused result %d\n", i); });
  // CHECK-MESSAGES-NOT: [[@LINE-1]]:6: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES-NOT: std::println("GCC statement expression with unused result {}", i);

  return fprintf(stderr, "Return value used in return\n");
}

void fprintf_simple() {
  fprintf(stderr, "Hello");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::print(stderr, "Hello");
}

void std_printf_simple() {
  std::printf("std::Hello");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("std::Hello");
}

void printf_escape() {
  printf("before \t");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("before \t");

  printf("\n after");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("\n after");

  printf("before \a after");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("before \a after");

  printf("Bell\a%dBackspace\bFF%s\fNewline\nCR\rTab\tVT\vEscape\x1b\x07%d", 42, "string", 99);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Bell\a{}Backspace\bFF{}\fNewline\nCR\rTab\tVT\vEscape\x1b\a{}", 42, "string", 99);

  printf("not special \x1b\x01\x7f");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("not special \x1b\x01\x7f");
}

void printf_percent() {
  printf("before %%");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("before %");

  printf("%% after");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("% after");

  printf("before %% after");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("before % after");

  printf("Hello %% and another %%");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Hello % and another %");

  printf("Not a string %%s");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Not a string %s");
}

void printf_curlies() {
  printf("%d {}", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{} {{[{][{]}}}}", 42);

  printf("{}");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{{[{][{]}}}}");
}

void printf_unsupported_format_specifiers() {
  int pos;
  printf("%d %n %d\n", 42, &pos, 72);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: unable to use 'std::println' instead of 'printf' because '%n' is not supported in format string [modernize-use-std-print]

  printf("Error %m\n");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: unable to use 'std::println' instead of 'printf' because '%m' is not supported in format string [modernize-use-std-print]
}

void printf_not_string_literal(const char *fmt) {
  // We can't convert the format string if it's not a literal
  printf(fmt, 42);
}

void printf_inttypes_ugliness() {
  // The one advantage of the checker seeing the token pasted version of the
  // format string is that we automatically cope with the horrendously-ugly
  // inttypes.h macros!
  int64_t u64 = 42;
  uintmax_t umax = 4242;
  printf("uint64:%" PRId64 " uintmax:%" PRIuMAX "\n", u64, umax);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("uint64:{} uintmax:{}", u64, umax);
}

void printf_raw_string() {
  // This one doesn't require the format string to be changed, so it stays intact
  printf(R"(First\Second)");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print(R"(First\Second)");

  // This one does require the format string to be changed, so unfortunately it
  // gets reformatted as a normal string.
  printf(R"(First %d\Second)", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("First {}\\Second", 42);
}

void printf_integer_d() {
  const bool b = true;
  // The "d" type is necessary here for compatibility with printf since
  // std::print will print booleans as "true" or "false".
  printf("Integer %d from bool\n", b);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {:d} from bool", b);

  // The "d" type is necessary here for compatibility with printf since
  // std::print will print booleans as "true" or "false".
  printf("Integer %i from bool\n", b);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {:d} from bool", b);

  // The 'd' is always necessary if we pass a char since otherwise the
  // parameter will be formatted as a character. In StrictMode, the
  // cast is always necessary to maintain the printf behaviour since
  // char may be unsigned, but this also means that the 'd' is not
  // necessary.
  const char c = 'A';
  printf("Integers %d %hhd from char\n", c, c);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integers {:d} {:d} from char", c, c);
  // CHECK-FIXES-STRICT: std::println("Integers {} {} from char", static_cast<signed char>(c), static_cast<signed char>(c));

  printf("Integers %i %hhi from char\n", c, c);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integers {:d} {:d} from char", c, c);
  // CHECK-FIXES-STRICT: std::println("Integers {} {} from char", static_cast<signed char>(c), static_cast<signed char>(c));

  const signed char sc = 'A';
  printf("Integers %d %hhd from signed char\n", sc, sc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integers {} {} from signed char", sc, sc);

  printf("Integers %i %hhi from signed char\n", sc, sc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integers {} {} from signed char", sc, sc);

  const unsigned char uc = 'A';
  printf("Integers %d %hhd from unsigned char\n", uc, uc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integers {} {} from unsigned char", uc, uc);
  // CHECK-FIXES-STRICT: std::println("Integers {} {} from unsigned char", static_cast<signed char>(uc), static_cast<signed char>(uc));

  printf("Integers %i %hhi from unsigned char\n", uc, uc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integers {} {} from unsigned char", uc, uc);
  // CHECK-FIXES-STRICT: std::println("Integers {} {} from unsigned char", static_cast<signed char>(uc), static_cast<signed char>(uc));

  const int8_t i8 = 42;
  printf("Integer %" PRIi8 " from int8_t\n", i8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from int8_t", i8);

  const int_fast8_t if8 = 42;
  printf("Integer %" PRIiFAST8 " from int_fast8_t\n", if8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from int_fast8_t", if8);

  const int_least8_t il8 = 42;
  printf("Integer %" PRIiFAST8 " from int_least8_t\n", il8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from int_least8_t", il8);

  const uint8_t u8 = 42U;
  const std::uint8_t su8 = u8;
  printf("Integers %" PRIi8 " and %" PRId8 " from uint8_t\n", u8, su8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integers {} and {} from uint8_t", u8, su8);
  // CHECK-FIXES-STRICT: std::println("Integers {} and {} from uint8_t", static_cast<int8_t>(u8), static_cast<std::int8_t>(su8));

  const uint_fast8_t uf8 = 42U;
  printf("Integer %" PRIiFAST8 " from uint_fast8_t\n", uf8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from uint_fast8_t", uf8);
  // CHECK-FIXES-STRICT: std::println("Integer {} from uint_fast8_t", static_cast<int_fast8_t>(uf8));

  const uint_least8_t ul8 = 42U;
  printf("Integer %" PRIiLEAST8 " from uint_least8_t\n", ul8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from uint_least8_t", ul8);
  // CHECK-FIXES-STRICT: std::println("Integer {} from uint_least8_t", static_cast<int_least8_t>(ul8));

  const short s = 42;
  printf("Integer %hd from short\n", s);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from short", s);

  printf("Integer %hi from short\n", s);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from short", s);

  const unsigned short us = 42U;
  printf("Integer %hd from unsigned short\n", us);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned short", us);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned short", static_cast<short>(us));

  printf("Integer %hi from unsigned short\n", us);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned short", us);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned short", static_cast<short>(us));

  const int i = 42;
  printf("Integer %d from integer\n", i);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from integer", i);

  printf("Integer %i from integer\n", i);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from integer", i);

  const unsigned int ui = 42U;
  printf("Integer %d from unsigned integer\n", ui);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned integer", ui);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned integer", static_cast<int>(ui));

  printf("Integer %i from unsigned integer\n", ui);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned integer", ui);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned integer", static_cast<int>(ui));

  const long l = 42L;
  printf("Integer %ld from long\n", l);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from long", l);

  printf("Integer %li from long\n", l);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from long", l);

  const unsigned long ul = 42UL;
  printf("Integer %ld from unsigned long\n", ul);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned long", ul);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned long", static_cast<long>(ul));

  printf("Integer %li from unsigned long\n", ul);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned long", ul);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned long", static_cast<long>(ul));

  const long long ll = 42LL;
  printf("Integer %lld from long long\n", ll);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from long long", ll);

  printf("Integer %lli from long long\n", ll);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integer {} from long long", ll);

  const unsigned long long ull = 42ULL;
  printf("Integer %lld from unsigned long long\n", ull);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned long long", ull);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned long long", static_cast<long long>(ull));

  printf("Integer %lli from unsigned long long\n", ull);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from unsigned long long", ull);
  // CHECK-FIXES-STRICT: std::println("Integer {} from unsigned long long", static_cast<long long>(ull));

  const intmax_t im = 42;
  const std::intmax_t sim = im;
  printf("Integers %jd and %jd from intmax_t\n", im, sim);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integers {} and {} from intmax_t", im, sim);

  printf("Integers %ji and %ji from intmax_t\n", im, sim);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integers {} and {} from intmax_t", im, sim);

  const uintmax_t uim = 42;
  const std::uintmax_t suim = uim;
  printf("Integers %jd and %jd from uintmax_t\n", uim, suim);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integers {} and {} from uintmax_t", uim, suim);
  // CHECK-FIXES-STRICT: std::println("Integers {} and {} from uintmax_t", static_cast<intmax_t>(uim), static_cast<std::intmax_t>(suim));

  printf("Integer %ji from intmax_t\n", uim);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integer {} from intmax_t", uim);
  // CHECK-FIXES-STRICT: std::println("Integer {} from intmax_t", static_cast<intmax_t>(uim));

  const int ai[] = { 0, 1, 2, 3};
  const ptrdiff_t pd = &ai[3] - &ai[0];
  const std::ptrdiff_t spd = pd;
  printf("Integers %td and %td from ptrdiff_t\n", pd, spd);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integers {} and {} from ptrdiff_t", pd, spd);

  printf("Integers %ti and %ti from ptrdiff_t\n", pd, spd);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Integers {} and {} from ptrdiff_t", pd, spd);

  const size_t z = 42UL;
  const std::size_t sz = z;
  printf("Integers %zd and %zd from size_t\n", z, sz);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Integers {} and {} from size_t", z, sz);
  // CHECK-FIXES-STRICT: std::println("Integers {} and {} from size_t", static_cast<ssize_t>(z), static_cast<std::ssize_t>(sz));
}

void printf_integer_u()
{
  const bool b = true;
  // The "d" type is necessary here for compatibility with printf since
  // std::print will print booleans as "true" or "false".
  printf("Unsigned integer %u from bool\n", b);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {:d} from bool", b);

  const char c = 'A';
  printf("Unsigned integer %hhu from char\n", c);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {:d} from char", c);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from char", static_cast<unsigned char>(c));

  const signed char sc = 'A';
  printf("Unsigned integer %hhu from signed char\n", sc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from signed char", sc);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from signed char", static_cast<unsigned char>(sc));

  printf("Unsigned integer %u from signed char\n", sc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from signed char", sc);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from signed char", static_cast<unsigned char>(sc));

  const unsigned char uc = 'A';
  printf("Unsigned integer %hhu from unsigned char\n", uc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from unsigned char", uc);

  printf("Unsigned integer %u from unsigned char\n", uc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from unsigned char", uc);

  const int8_t i8 = 42;
  printf("Unsigned integer %" PRIu8 " from int8_t\n", i8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from int8_t", i8);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from int8_t", static_cast<uint8_t>(i8));

  const int_fast8_t if8 = 42;
  printf("Unsigned integer %" PRIuFAST8 " from int_fast8_t\n", if8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from int_fast8_t", if8);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from int_fast8_t", static_cast<uint_fast8_t>(if8));

  const int_least8_t il8 = 42;
  printf("Unsigned integer %" PRIuFAST8 " from int_least8_t\n", il8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from int_least8_t", il8);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from int_least8_t", static_cast<uint_least8_t>(il8));

  const uint8_t u8 = 42U;
  printf("Unsigned integer %" PRIu8 " from uint8_t\n", u8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from uint8_t", u8);

  const uint_fast8_t uf8 = 42U;
  printf("Unsigned integer %" PRIuFAST8 " from uint_fast8_t\n", uf8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from uint_fast8_t", uf8);

  const uint_least8_t ul8 = 42U;
  printf("Unsigned integer %" PRIuLEAST8 " from uint_least8_t\n", ul8);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from uint_least8_t", ul8);

  const short s = 42;
  printf("Unsigned integer %hu from short\n", s);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from short", s);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from short", static_cast<unsigned short>(s));

  const unsigned short us = 42U;
  printf("Unsigned integer %hu from unsigned short\n", us);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from unsigned short", us);

  const int i = 42;
  printf("Unsigned integer %u from signed integer\n", i);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from signed integer", i);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from signed integer", static_cast<unsigned int>(i));

  const unsigned int ui = 42U;
  printf("Unsigned integer %u from unsigned integer\n", ui);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from unsigned integer", ui);

  const long l = 42L;
  printf("Unsigned integer %u from signed long\n", l);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from signed long", l);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from signed long", static_cast<unsigned long>(l));

  const unsigned long ul = 42UL;
  printf("Unsigned integer %lu from unsigned long\n", ul);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from unsigned long", ul);

  const long long ll = 42LL;
  printf("Unsigned integer %llu from long long\n", ll);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integer {} from long long", ll);
  // CHECK-FIXES-STRICT: std::println("Unsigned integer {} from long long", static_cast<unsigned long long>(ll));

  const unsigned long long ull = 42ULL;
  printf("Unsigned integer %llu from unsigned long long\n", ull);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integer {} from unsigned long long", ull);

  const intmax_t im = 42;
  const std::intmax_t sim = im;
  printf("Unsigned integers %ju and %ju from intmax_t\n", im, sim);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integers {} and {} from intmax_t", im, sim);
  // CHECK-FIXES-STRICT: std::println("Unsigned integers {} and {} from intmax_t", static_cast<uintmax_t>(im), static_cast<std::uintmax_t>(sim));

  const uintmax_t uim = 42U;
  const std::uintmax_t suim = uim;
  printf("Unsigned integers %ju and %ju from uintmax_t\n", uim, suim);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integers {} and {} from uintmax_t", uim, suim);

  const int ai[] = { 0, 1, 2, 3};
  const ptrdiff_t pd = &ai[3] - &ai[0];
  const std::ptrdiff_t spd = pd;
  printf("Unsigned integers %tu and %tu from ptrdiff_t\n", pd, spd);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println("Unsigned integers {} and {} from ptrdiff_t", pd, spd);
  // CHECK-FIXES-STRICT: std::println("Unsigned integers {} and {} from ptrdiff_t", static_cast<size_t>(pd), static_cast<std::size_t>(spd));

  const size_t z = 42U;
  const std::size_t sz = z;
  printf("Unsigned integers %zu and %zu from size_t\n", z, sz);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Unsigned integers {} and {} from size_t", z, sz);
}

// This checks that we get the argument offset right with the extra FILE * argument
void fprintf_integer() {
  fprintf(stderr, "Integer %d from integer\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "Integer {} from integer", 42);

  fprintf(stderr, "Integer %i from integer\n", 65);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "Integer {} from integer", 65);

  fprintf(stderr, "Integer %i from char\n", 'A');
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println(stderr, "Integer {:d} from char", 'A');
  // CHECK-FIXES-STRICT: std::println(stderr, "Integer {} from char", static_cast<signed char>('A'));

  fprintf(stderr, "Integer %d from char\n", 'A');
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES-NOTSTRICT: std::println(stderr, "Integer {:d} from char", 'A');
  // CHECK-FIXES-STRICT: std::println(stderr, "Integer {} from char", static_cast<signed char>('A'));
}

void printf_char() {
  const char c = 'A';
  printf("Char %c from char\n", c);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {} from char", c);

  const signed char sc = 'A';
  printf("Char %c from signed char\n", sc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {:c} from signed char", sc);

  const unsigned char uc = 'A';
  printf("Char %c from unsigned char\n", uc);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {:c} from unsigned char", uc);

  const int i = 65;
  printf("Char %c from integer\n", i);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {:c} from integer", i);

  const unsigned int ui = 65;
  printf("Char %c from unsigned integer\n", ui);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {:c} from unsigned integer", ui);

  const unsigned long long ull = 65;
  printf("Char %c from unsigned long long\n", ull);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {:c} from unsigned long long", ull);
}

void printf_bases() {
  printf("Hex %lx\n", 42L);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hex {:x}", 42L);

  printf("HEX %X\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("HEX {:X}", 42);

  printf("Oct %lo\n", 42L);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Oct {:o}", 42L);
}

void printf_alternative_forms() {
  printf("Hex %#lx\n", 42L);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hex {:#x}", 42L);

  printf("HEX %#X\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("HEX {:#X}", 42);

  printf("Oct %#lo\n", 42L);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Oct {:#o}", 42L);

  printf("Double %#f %#F\n", -42.0, -42.0);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Double {:#f} {:#F}", -42.0, -42.0);

  printf("Double %#g %#G\n", -42.0, -42.0);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Double {:#g} {:#G}", -42.0, -42.0);

  printf("Double %#e %#E\n", -42.0, -42.0);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Double {:#e} {:#E}", -42.0, -42.0);

  printf("Double %#a %#A\n", -42.0, -42.0);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Double {:#a} {:#A}", -42.0, -42.0);

  // Characters don't have an alternate form
  printf("Char %#c\n", 'A');
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {}", 'A');

  // Strings don't have an alternate form
  printf("Char %#c\n", 'A');
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Char {}", 'A');
}

void printf_string() {
  printf("Hello %s after\n", "Goodbye");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {} after", "Goodbye");

  // std::print can't print signed char strings.
  const signed char *sstring = reinterpret_cast<const signed char *>("ustring");
  printf("signed char string %s\n", sstring);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("signed char string {}", reinterpret_cast<const char *>(sstring));

  // std::print can't print unsigned char strings.
  const unsigned char *ustring = reinterpret_cast<const unsigned char *>("ustring");
  printf("unsigned char string %s\n", ustring);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("unsigned char string {}", reinterpret_cast<const char *>(ustring));
}

void printf_float() {
  // If the type is not specified then either f or e will be used depending on
  // whichever is shorter. This means that it is necessary to be specific to
  // maintain compatibility with printf.

  // TODO: Should we force a cast here, since printf will promote to double
  // automatically, but std::format will not, which could result in different
  // output?

  const float f = 42.0F;
  printf("Hello %f after\n", f);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:f} after", f);

  printf("Hello %g after\n", f);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:g} after", f);

  printf("Hello %e after\n", f);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:e} after", f);
}

void printf_double() {
  // If the type is not specified then either f or e will be used depending on
  // whichever is shorter. This means that it is necessary to be specific to
  // maintain compatibility with printf.

  const double d = 42.0;
  printf("Hello %f after\n", d);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:f} after", d);

  printf("Hello %g after\n", d);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:g} after", d);

  printf("Hello %e after\n", d);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:e} after", d);
}

void printf_long_double() {
  // If the type is not specified then either f or e will be used depending on
  // whichever is shorter. This means that it is necessary to be specific to
  // maintain compatibility with printf.

  const long double ld = 42.0L;
  printf("Hello %Lf after\n", ld);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:f} after", ld);

  printf("Hello %g after\n", ld);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:g} after", ld);

  printf("Hello %e after\n", ld);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:e} after", ld);
}

void printf_pointer() {
  int i;
  double j;
  printf("Int* %p %s %p\n", &i, "Double*", &j);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Int* {} {} {}", static_cast<const void *>(&i), "Double*", static_cast<const void *>(&j));

  printf("%p\n", nullptr);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", nullptr);

  const auto np = nullptr;
  printf("%p\n", np);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", np);

  // NULL isn't a pointer, so std::print needs some help.
  printf("%p\n", NULL);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", static_cast<const void *>(NULL));

  printf("%p\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", static_cast<const void *>(42));

  // If we already have a void pointer then no cast is required.
  printf("%p\n", reinterpret_cast<const void *>(44));
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", reinterpret_cast<const void *>(44));

  const void *p;
  printf("%p\n", p);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", p);

  // But a pointer to a pointer to void does need a cast
  const void **pp;
  printf("%p\n", pp);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", static_cast<const void *>(pp));

  printf("%p\n", printf_pointer);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("{}", static_cast<const void *>(printf_pointer));
}

class AClass
{
  int member;

  void printf_this_pointer()
  {
    printf("%p\n", this);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("{}", static_cast<const void *>(this));
  }

  void printf_pointer_to_member_function()
  {
    printf("%p\n", &AClass::printf_pointer_to_member_function);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("{}", static_cast<const void *>(&AClass::printf_pointer_to_member_function));
  }

  void printf_pointer_to_member_variable()
  {
    printf("%p\n", &AClass::member);
    // CHECK-MESSAGES: [[@LINE-1]]:5: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
    // CHECK-FIXES: std::println("{}", static_cast<const void *>(&AClass::member));
  }
};

void printf_positional_arg() {
  printf("%1$d", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{0}", 42);

  printf("before %1$d", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("before {0}", 42);

  printf("%1$d after", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{0} after", 42);

  printf("before %1$d after", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("before {0} after", 42);

  printf("before %2$d between %1$s after", "string", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("before {1} between {0} after", "string", 42);
}

// printf always defaults to right justification,, no matter what the type is of
// the argument. std::format uses left justification by default for strings, and
// right justification for numbers.
void printf_right_justified() {
  printf("Right-justified integer %4d after\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Right-justified integer {:4} after", 42);

  printf("Right-justified double %4f\n", 227.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Right-justified double {:4f}", 227.2);

  printf("Right-justified double %4g\n", 227.4);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Right-justified double {:4g}", 227.4);

  printf("Right-justified integer with field width argument %*d after\n", 5, 424242);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Right-justified integer with field width argument {:{}} after", 424242, 5);

  printf("Right-justified integer with field width argument %2$*1$d after\n", 5, 424242);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Right-justified integer with field width argument {1:{0}} after", 5, 424242);

  printf("Right-justified string %20s\n", "Hello");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Right-justified string {:>20}", "Hello");

  printf("Right-justified string with field width argument %2$*1$s after\n", 20, "wibble");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Right-justified string with field width argument {1:>{0}} after", 20, "wibble");
}

// printf always requires - for left justification, no matter what the type is
// of the argument. std::format uses left justification by default for strings,
// and right justification for numbers.
void printf_left_justified() {
  printf("Left-justified integer %-4d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified integer {:<4}", 42);

  printf("Left-justified integer %--4d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified integer {:<4}", 42);

  printf("Left-justified double %-4f\n", 227.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified double {:<4f}", 227.2);

  printf("Left-justified double %-4g\n", 227.4);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified double {:<4g}", 227.4);

  printf("Left-justified integer with field width argument %-*d after\n", 5, 424242);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified integer with field width argument {:<{}} after", 424242, 5);

  printf("Left-justified integer with field width argument %2$-*1$d after\n", 5, 424242);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified integer with field width argument {1:<{0}} after", 5, 424242);

  printf("Left-justified string %-20s\n", "Hello");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified string {:20}", "Hello");

  printf("Left-justified string with field width argument %2$-*1$s after\n", 5, "wibble");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Left-justified string with field width argument {1:{0}} after", 5, "wibble");
}

void printf_precision() {
  printf("Hello %.3f\n", 3.14159);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:.3f}", 3.14159);

  printf("Hello %10.3f\n", 3.14159);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:10.3f}", 3.14159);

  printf("Hello %.*f after\n", 10, 3.14159265358979323846);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:.{}f} after", 3.14159265358979323846, 10);

  printf("Hello %10.*f after\n", 3, 3.14159265358979323846);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:10.{}f} after", 3.14159265358979323846, 3);

  printf("Hello %*.*f after\n", 10, 4, 3.14159265358979323846);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:{}.{}f} after", 3.14159265358979323846, 10, 4);

  printf("Hello %1$.*2$f after\n", 3.14159265358979323846, 4);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {0:.{1}f} after", 3.14159265358979323846, 4);

  // Precision is ignored, but maintained on non-numeric arguments
  printf("Hello %.5s\n", "Goodbye");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:.5}", "Goodbye");

  printf("Hello %.5c\n", 'G');
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Hello {:.5}", 'G');
}

void printf_field_width_and_precision() {
  printf("width only:%*d width and precision:%*.*f precision only:%.*f\n", 3, 42, 4, 2, 3.14159265358979323846, 5, 2.718);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("width only:{:{}} width and precision:{:{}.{}f} precision only:{:.{}f}", 42, 3, 3.14159265358979323846, 4, 2, 2.718, 5);

  printf("width and precision positional:%1$*2$.*3$f after\n", 3.14159265358979323846, 4, 2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("width and precision positional:{0:{1}.{2}f} after", 3.14159265358979323846, 4, 2);

  const int width = 10, precision = 3;
  printf("width only:%3$*1$d width and precision:%4$*1$.*2$f precision only:%5$.*2$f\n", width, precision, 42, 3.1415926, 2.718);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("width only:{2:{0}} width and precision:{3:{0}.{1}f} precision only:{4:.{1}f}", width, precision, 42, 3.1415926, 2.718);
}

void fprintf_field_width_and_precision() {
  fprintf(stderr, "width only:%*d width and precision:%*.*f precision only:%.*f\n", 3, 42, 4, 2, 3.14159265358979323846, 5, 2.718);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "width only:{:{}} width and precision:{:{}.{}f} precision only:{:.{}f}", 42, 3, 3.14159265358979323846, 4, 2, 2.718, 5);

  fprintf(stderr, "width and precision positional:%1$*2$.*3$f after\n", 3.14159265358979323846, 4, 2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "width and precision positional:{0:{1}.{2}f} after", 3.14159265358979323846, 4, 2);

  const int width = 10, precision = 3;
  fprintf(stderr, "width only:%3$*1$d width and precision:%4$*1$.*2$f precision only:%5$.*2$f\n", width, precision, 42, 3.1415926, 2.718);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::println(stderr, "width only:{2:{0}} width and precision:{3:{0}.{1}f} precision only:{4:.{1}f}", width, precision, 42, 3.1415926, 2.718);
}

void printf_alternative_form() {
  printf("Wibble %#x\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Wibble {:#x}", 42);

  printf("Wibble %#20x\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Wibble {:#20x}", 42);

  printf("Wibble %#020x\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Wibble {:#020x}", 42);

  printf("Wibble %#-20x\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Wibble {:<#20x}", 42);
}

void printf_leading_plus() {
  printf("Positive integer %+d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Positive integer {:+}", 42);

  printf("Positive double %+f\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Positive double {:+f}", 42.2);

  printf("Positive double %+g\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Positive double {:+g}", 42.2);

  // Ignore leading plus on strings to avoid potential runtime exception where
  // printf would have just ignored it.
  printf("Positive string %+s\n", "string");
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Positive string {}", "string");
}

void printf_leading_space() {
  printf("Spaced integer % d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Spaced integer {: }", 42);

  printf("Spaced integer %- d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Spaced integer {: }", 42);

  printf("Spaced double % f\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Spaced double {: f}", 42.2);

  printf("Spaced double % g\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Spaced double {: g}", 42.2);
}

void printf_leading_zero() {
  printf("Leading zero integer %03d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero integer {:03}", 42);

  printf("Leading minus and zero integer %-03d minus ignored\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading minus and zero integer {:<03} minus ignored", 42);

  printf("Leading zero unsigned integer %03u\n", 42U);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero unsigned integer {:03}", 42U);

  printf("Leading zero double %03f\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero double {:03f}", 42.2);

  printf("Leading zero double %03g\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero double {:03g}", 42.2);
}

void printf_leading_plus_and_space() {
  // printf prefers plus to space. {fmt} will throw if both are present.
  printf("Spaced integer % +d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Spaced integer {:+}", 42);

  printf("Spaced double %+ f\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Spaced double {:+f}", 42.2);

  printf("Spaced double % +g\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Spaced double {:+g}", 42.2);
}

void printf_leading_zero_and_plus() {
  printf("Leading zero integer %+03d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero integer {:+03}", 42);

  printf("Leading zero double %0+3f\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero double {:+03f}", 42.2);

  printf("Leading zero double %0+3g\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero double {:+03g}", 42.2);
}

void printf_leading_zero_and_space() {
  printf("Leading zero and space integer %0 3d\n", 42);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero and space integer {: 03}", 42);

  printf("Leading zero and space double %0 3f\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero and space double {: 03f}", 42.2);

  printf("Leading zero and space double %0 3g\n", 42.2);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("Leading zero and space double {: 03g}", 42.2);
}

// add signed plained enum too
enum PlainEnum { red };
enum SignedPlainEnum { black = -42 };
enum BoolEnum : unsigned int { yellow };
enum CharEnum : char { purple };
enum SCharEnum : signed char  { aquamarine };
enum UCharEnum : unsigned char  { pink };
enum ShortEnum : short { beige };
enum UShortEnum : unsigned short { grey };
enum IntEnum : int { green };
enum UIntEnum : unsigned int { blue };
enum LongEnum : long { magenta };
enum ULongEnum : unsigned long { cyan };
enum LongLongEnum : long long { taupe };
enum ULongLongEnum : unsigned long long { brown };

void printf_enum_d() {
  PlainEnum plain_enum;
  printf("%d", plain_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<int>(plain_enum));

  SignedPlainEnum splain_enum;
  printf("%d", splain_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<int>(splain_enum));

  BoolEnum bool_enum;
  printf("%d", bool_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<int>(bool_enum));

  CharEnum char_enum;
  printf("%d", char_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<signed char>(char_enum));

  SCharEnum schar_enum;
  printf("%d", schar_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<signed char>(schar_enum));

  UCharEnum uchar_enum;
  printf("%d", uchar_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<signed char>(uchar_enum));

  ShortEnum short_enum;
  printf("%d", short_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<short>(short_enum));

  UShortEnum ushort_enum;
  printf("%d", ushort_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<short>(ushort_enum));

  IntEnum int_enum;
  printf("%d", int_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<int>(int_enum));

  UIntEnum uint_enum;
  printf("%d", uint_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<int>(uint_enum));

  LongEnum long_enum;
  printf("%d", long_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<long>(long_enum));

  ULongEnum ulong_enum;
  printf("%d", ulong_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<long>(ulong_enum));

  LongLongEnum longlong_enum;
  printf("%d", longlong_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<long long>(longlong_enum));

  ULongLongEnum ulonglong_enum;
  printf("%d", ulonglong_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<long long>(ulonglong_enum));
}

void printf_enum_u() {
  PlainEnum plain_enum;
  printf("%u", plain_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned int>(plain_enum));

  SignedPlainEnum splain_enum;
  printf("%u", splain_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned int>(splain_enum));

  BoolEnum bool_enum;
  printf("%u", bool_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned int>(bool_enum));

  CharEnum char_enum;
  printf("%u", char_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned char>(char_enum));

  SCharEnum schar_enum;
  printf("%u", schar_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned char>(schar_enum));

  UCharEnum uchar_enum;
  printf("%u", uchar_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned char>(uchar_enum));

  ShortEnum short_enum;
  printf("%u", short_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned short>(short_enum));

  UShortEnum ushort_enum;
  printf("%u", ushort_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned short>(ushort_enum));

  IntEnum int_enum;
  printf("%u", int_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned int>(int_enum));

  UIntEnum uint_enum;
  printf("%u", uint_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned int>(uint_enum));

  LongEnum long_enum;
  printf("%u", long_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned long>(long_enum));

  ULongEnum ulong_enum;
  printf("%u", ulong_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned long>(ulong_enum));

  LongLongEnum longlong_enum;
  printf("%u", longlong_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned long long>(longlong_enum));

  ULongLongEnum ulonglong_enum;
  printf("%u", ulonglong_enum);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("{}", static_cast<unsigned long long>(ulonglong_enum));
}

void printf_string_function(const char *(*callback)()) {
  printf("printf string from callback %s", callback());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("printf string from callback {}", callback());
}

template <typename CharType>
struct X
{
  const CharType *str() const;
};

void printf_string_member_function(const X<char> &x, const X<const char> &cx) {
  printf("printf string from member function %s", x.str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("printf string from member function {}", x.str());

  printf("printf string from member function on const %s", cx.str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("printf string from member function on const {}", cx.str());
}

void printf_string_cstr(const std::string &s1, const std::string &s2) {
  printf("printf string one c_str %s", s1.c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("printf string one c_str {}", s1);

  printf("printf string two c_str %s %s\n", s1.c_str(), s2.data());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("printf string two c_str {} {}", s1, s2);
}

void printf_not_char_string_cstr(const std::wstring &ws1) {
  // This test is to check that we only remove
  // std::basic_string<CharType>::c_str()/data() when CharType is char. I've
  // been unable to come up with a genuine situation where someone would have
  // actually successfully called those methods when this isn't the case without
  // -Wformat warning, but it seems sensible to restrict removal regardless.
  printf("printf bogus wstring c_str %s", ws1.c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("printf bogus wstring c_str {}", ws1.c_str());
}

void fprintf_string_cstr(const std::string &s1) {
  fprintf(stderr, "fprintf string c_str %s", s1.c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::print(stderr, "fprintf string c_str {}", s1);
}

void printf_string_pointer_cstr(const std::string *s1, const std::string *s2) {
  printf("printf string pointer one c_str %s", s1->c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("printf string pointer one c_str {}", *s1);

  printf("printf string pointer two c_str %s %s\n", s1->c_str(), s2->data());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("printf string pointer two c_str {} {}", *s1, *s2);
}

void fprintf_string_pointer_cstr(const std::string *s1) {
  fprintf(stderr, "fprintf string pointer c_str %s", s1->c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'fprintf' [modernize-use-std-print]
  // CHECK-FIXES: std::print(stderr, "fprintf string pointer c_str {}", *s1);
}

template <typename T>
struct iterator {
  T *operator->();
  T &operator*();
};

void printf_iterator_cstr(iterator<std::string> i1, iterator<std::string> i2)
{
  printf("printf iterator c_str %s %s\n", i1->c_str(), i2->data());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::println' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::println("printf iterator c_str {} {}", *i1, *i2);
}

// Something that isn't std::string, so the calls to c_str() and data() must not
// be removed even though the printf call will be replaced.
struct S
{
  const char *c_str() const;
  const char *data() const;
};

void p(S s1, S *s2)
{
  printf("Not std::string %s %s", s1.c_str(), s2->c_str());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Not std::string {} {}", s1.c_str(), s2->c_str());

  printf("Not std::string %s %s", s1.data(), s2->data());
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: use 'std::print' instead of 'printf' [modernize-use-std-print]
  // CHECK-FIXES: std::print("Not std::string {} {}", s1.data(), s2->data());
}
