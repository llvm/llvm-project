// RUN: %clang_analyze_cc1 -w -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=unix.cstring.NullArg \
// RUN:   -analyzer-checker=alpha.unix.cstring \
// RUN:   -analyzer-checker=debug.ExprInspection

#define NULL ((void *)0)

typedef __typeof(sizeof(int)) size_t;
size_t strlcpy(char *dst, const char *src, size_t n);
size_t strlcat(char *dst, const char *src, size_t n);
size_t strlen(const char *s);
void clang_analyzer_eval(int);

void f1(void) {
  char overlap[] = "123456789";
  strlcpy(overlap, overlap + 1, 3); // expected-warning{{Arguments must not be overlapping buffers}}
}

void f2(void) {
  char buf[5];
  size_t len;
  len = strlcpy(buf, "abcd", sizeof(buf)); // expected-no-warning
  clang_analyzer_eval(len == 4); // expected-warning{{TRUE}}
  len = strlcat(buf, "efgh", sizeof(buf)); // expected-no-warning
  clang_analyzer_eval(len == 8); // expected-warning{{TRUE}}
}

void f3(void) {
  char dst[2];
  const char *src = "abdef";
  strlcpy(dst, src, 5); // expected-warning{{String copy function overflows the destination buffer}}
}

void f4(void) {
  strlcpy(NULL, "abcdef", 6); // expected-warning{{Null pointer passed as 1st argument to string copy function}}
}

void f5(void) {
  strlcat(NULL, "abcdef", 6); // expected-warning{{Null pointer passed as 1st argument to string concatenation function}}
}

void f6(void) {
  char buf[8];
  strlcpy(buf, "abc", 3);
  size_t len = strlcat(buf, "defg", 4);
  clang_analyzer_eval(len == 7); // expected-warning{{TRUE}}
}

int f7(void) {
  char buf[8];
  return strlcpy(buf, "1234567", 0); // no-crash
}

void f8(void){
  char buf[5];
  size_t len;

  // basic strlcpy
  len = strlcpy(buf,"123", sizeof(buf));
  clang_analyzer_eval(len==3);// expected-warning{{TRUE}}
  len = strlen(buf);
  clang_analyzer_eval(len==3);// expected-warning{{TRUE}}

  // testing bounded strlcat
  len = strlcat(buf,"456", sizeof(buf));
  clang_analyzer_eval(len==6);// expected-warning{{TRUE}}
  len = strlen(buf);
  clang_analyzer_eval(len==4);// expected-warning{{TRUE}}

  // testing strlcat with size==0
  len = strlcat(buf,"789", 0);
  clang_analyzer_eval(len==7);// expected-warning{{TRUE}}
  len = strlen(buf);
  clang_analyzer_eval(len==4);// expected-warning{{TRUE}}

  // testing strlcpy with size==0
  len = strlcpy(buf,"123",0);
  clang_analyzer_eval(len==3);// expected-warning{{TRUE}}
  len = strlen(buf);
  clang_analyzer_eval(len==4);// expected-warning{{TRUE}}

}

void f9(int unknown_size, char* unknown_src, char* unknown_dst){
  char buf[8];
  size_t len;

  len = strlcpy(buf,"abba",sizeof(buf));

  clang_analyzer_eval(len==4);// expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(buf)==4);// expected-warning{{TRUE}}

  //size is unknown
  len = strlcat(buf,"cd", unknown_size);
  clang_analyzer_eval(len==6);// expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(buf)>=4);// expected-warning{{TRUE}}

  //dst is unknown
  len = strlcpy(unknown_dst,"abbc",unknown_size);
  clang_analyzer_eval(len==4);// expected-warning{{TRUE}}
  clang_analyzer_eval(strlen(unknown_dst));// expected-warning{{UNKNOWN}}

  //src is unknown
  len = strlcpy(buf,unknown_src, sizeof(buf));
  clang_analyzer_eval(len);// expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen(buf));// expected-warning{{UNKNOWN}}

  //src, dst is unknown
  len = strlcpy(unknown_dst, unknown_src, unknown_size);
  clang_analyzer_eval(len);// expected-warning{{UNKNOWN}}
  clang_analyzer_eval(strlen(unknown_dst));// expected-warning{{UNKNOWN}}

  //size is unknown
  len = strlcat(buf + 2, unknown_src + 1, sizeof(buf));
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}

void f10(void){
  char buf[8];
  size_t len;

  len = strlcpy(buf,"abba",sizeof(buf));
  clang_analyzer_eval(len==4);// expected-warning{{TRUE}}
  strlcat(buf, "efghi", 9);
  // expected-warning@-1 {{String concatenation function overflows the destination buffer}}
}

void f11(void) {
  //test for Bug 41729
  char a[256], b[256];
  strlcpy(a, "world", sizeof(a));
  strlcpy(b, "hello ", sizeof(b));
  strlcat(b, a, sizeof(b)); // no-warning
}

int a, b;
void unknown_val_crash(void) {
  // We're unable to evaluate the integer-to-pointer cast.
  strlcat(&b, a, 0); // no-crash
}
