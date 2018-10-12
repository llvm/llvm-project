// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -std=c11

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1

typedef unsigned int size_t;
typedef int errno_t;
size_t strlen(const char *);
void *malloc(size_t);
void *realloc(void *, size_t);

errno_t strncpy_s(char *, size_t, const char *, size_t);
errno_t strcpy_s(char *, size_t, const char *);
char *strcpy(char *, const char *);

errno_t memcpy_s(void *, size_t, const void *, size_t);
void *memcpy(void *, const void *, size_t);

#define SRC_LENGTH 3
#define SRC "foo"


void good_memcpy_known_src() {
  char dest[13];
  char src[] = "foobar";
  memcpy(dest, src, sizeof(src));
}

void good_memcpy_null_terminated(const char *src) {
  char dest[13];
  const int length = strlen(src);
  memcpy(dest, src, length);
  dest[length] = '\0';
}

void good_memcpy_proper_length(const char *src) {
  char *dest = 0;
  int length = strlen(src) + 1;
  dest = (char *)malloc(length);
  memcpy(dest, src, length);
}

void may_bad_memcpy_unknown_length(const char *src, int length) {
  char dest[13];
  memcpy(dest, src, length);
}

void may_bad_memcpy_const_length(const char *src) {
  char dest[13];
  memcpy(dest, src, 12);
}

void bad_memcpy_unknown_dest(char *dest01, const char *src) {
  memcpy(dest01, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strcpy(dest01, src);
}

void good_memcpy_unknown_dest(char *dst01, const char *src) {
  strcpy(dst01, src);
}

void bad_memcpy_variable_array(int dest_length) {
  char dest02[dest_length + 1];
  memcpy(dest02, "foobarbazqux", strlen("foobarbazqux"));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strcpy(dest02, "foobarbazqux");
}

void good_memcpy_variable_array(int dest_length) {
  char dst02[dest_length + 1];
  strcpy(dst02, "foobarbazqux");
}

void bad_memcpy_equal_src_length_and_length() {
  char dest03[13];
  const char *src = "foobarbazqux";
  memcpy(dest03, src, 12);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strcpy(dest03, src);
}

void good_memcpy_equal_src_length_and_length() {
  char dst03[13];
  const char *src = "foobarbazqux";
  strcpy(dst03, src);
}

void bad_memcpy_dest_size_overflows(const char *src) {
  const int length = strlen(src);
  char *dest04 = (char *)malloc(length);
  memcpy(dest04, src, length);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char *dest04 = (char *)malloc(length + 1);
  // CHECK-FIXES-NEXT: strcpy(dest04, src);
}

void good_memcpy_dest_size_overflows(const char *src) {
  const int length = strlen(src);
  char *dst04 = (char *)malloc(length + 1);
  strcpy(dst04, src);
}

void bad_memcpy_macro() {
  unsigned char dest05[13];
  memcpy(dest05, SRC, SRC_LENGTH);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: memcpy_s(dest05, 13, SRC, SRC_LENGTH + 1);
}

void good_memcpy_macro() {
  unsigned char dst05[13];
  memcpy_s(dst05, 13, SRC, SRC_LENGTH + 1);
}
