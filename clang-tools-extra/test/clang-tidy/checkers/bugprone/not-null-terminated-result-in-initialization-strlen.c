// RUN: %check_clang_tidy --match-partial-fixes %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -I %S/Inputs/not-null-terminated-result

#include "not-null-terminated-result-c.h"

void path_sensitive_unknown_length(char *position, const char *src) {
  int length;
  length = strlen(src);
  position = (char *)memchr(src, '\0', length);
}

void bad_memchr(char *position, const char *src) {
  int length = strlen(src);
  position = (char *)memchr(src, '\0', length);
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: the length is too short to include the null terminator [bugprone-not-null-terminated-result]
  // CHECK-FIXES: position = strchr(src, '\0');
}

void good_memchr(char *pos, const char *src) {
  pos = strchr(src, '\0');
}

void bad_strerror_s(int errno) {
  char dest[13];
  int length = strlen(strerror(errno));
  strerror_s(dest, length, errno);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'strerror_s' is not null-terminated and missing the last character of the error message [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest[14];
  // CHECK-FIXES-NEXT: int length = strlen(strerror(errno));
  // CHECK-FIXES-NEXT: strerror_s(dest, length + 1, errno);
}

void good_strerror_s(int errno) {
  char dst[14];
  int length = strlen(strerror(errno));
  strerror_s(dst, length + 1, errno);
}

int bad_strncmp_1(char *str1, const char *str2) {
  int length = strlen(str1) + 1;
  return strncmp(str1, str2, length);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strncmp(str1, str2, length - 1);
}

int good_strncmp_1(char *str1, const char *str2) {
  int length = strlen(str1) + 1;
  return strncmp(str1, str2, length - 1);
}

int bad_strncmp_2(char *str2) {
  return strncmp(str2, "foobar", (strlen("foobar") + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strncmp(str2, "foobar", (strlen("foobar")));
}

int bad_strncmp_3(char *str3) {
  return strncmp(str3, "foobar", 1 + strlen("foobar"));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strncmp(str3, "foobar", strlen("foobar"));
}

int good_strncmp_2_3(char *str) {
  return strncmp(str, "foobar", strlen("foobar"));
}

void bad_strxfrm(const char *long_source_name) {
  char long_destination_name[13];
  int very_long_length_definition_name = strlen(long_source_name);
  strxfrm(long_destination_name, long_source_name,
          very_long_length_definition_name);
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: the result from calling 'strxfrm' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char long_destination_name[14];
  // CHECK-FIXES-NEXT: int very_long_length_definition_name = strlen(long_source_name);
  // CHECK-FIXES-NEXT: strxfrm(long_destination_name, long_source_name,
  // CHECK-FIXES-NEXT: very_long_length_definition_name + 1);
}

void good_strxfrm(const char *long_source_name) {
  char long_destination_name[14];
  int very_long_length_definition_name = strlen(long_source_name);
  strxfrm(long_destination_name, long_source_name,
          very_long_length_definition_name + 1);
}
