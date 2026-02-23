// RUN: %check_clang_tidy %s bugprone-sprintf-to-snprintf %t

extern "C" int sprintf(char *str, const char *format, ...);
extern "C" int snprintf(char *s, unsigned long n, const char *format, ...);

void f() {
  char buff[80];
  sprintf(buff, "Hello, %s!\n", "world");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use 'snprintf' instead of 'sprintf' for fixed-size character arrays [bugprone-sprintf-to-snprintf]
  // CHECK-FIXES: snprintf(buff, sizeof(buff), "Hello, %s!\n", "world");
}

void ignore_pointers(char* ptr) {
  sprintf(ptr, "Hello");
  // Should not trigger because it's not a fixed-size array.
}