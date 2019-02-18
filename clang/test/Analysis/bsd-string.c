// RUN: %clang_analyze_cc1 -analyzer-checker=core,unix.cstring.NullArg,alpha.unix.cstring,debug.ExprInspection -analyzer-store=region -verify %s

#define NULL ((void *)0)

typedef __typeof(sizeof(int)) size_t;
size_t strlcpy(char *dst, const char *src, size_t n);
size_t strlcat(char *dst, const char *src, size_t n);
void clang_analyzer_eval(int);

void f1() {
  char overlap[] = "123456789";
  strlcpy(overlap, overlap + 1, 3); // expected-warning{{Arguments must not be overlapping buffers}}
}

void f2() {
  char buf[5];
  strlcpy(buf, "abcd", sizeof(buf)); // expected-no-warning
  strlcat(buf, "efgh", sizeof(buf)); // expected-warning{{Size argument is greater than the free space in the destination buffer}}
}

void f3() {
  char dst[2];
  const char *src = "abdef";
  strlcpy(dst, src, 5); // expected-warning{{Size argument is greater than the length of the destination buffer}}
}

void f4() {
  strlcpy(NULL, "abcdef", 6); // expected-warning{{Null pointer argument in call to string copy function}}
}

void f5() {
  strlcat(NULL, "abcdef", 6); // expected-warning{{Null pointer argument in call to string copy function}}
}

void f6() {
  char buf[8];
  strlcpy(buf, "abc", 3);
  size_t len = strlcat(buf, "defg", 4);
  clang_analyzer_eval(len == 7); // expected-warning{{TRUE}}
}

int f7() {
  char buf[8];
  return strlcpy(buf, "1234567", 0); // no-crash
}
