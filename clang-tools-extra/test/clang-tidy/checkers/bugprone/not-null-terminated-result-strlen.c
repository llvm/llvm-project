// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -I %S/Inputs/not-null-terminated-result

// FIXME: Something wrong with the APInt un/signed conversion on Windows:
// in 'strncmp(str6, "string", 7);' it tries to inject '4294967302' as length.

// UNSUPPORTED: system-windows

#include "not-null-terminated-result-c.h"

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1

void bad_memchr_1(char *position, const char *src) {
  position = (char *)memchr(src, '\0', strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:40: warning: the length is too short to include the null terminator [bugprone-not-null-terminated-result]
  // CHECK-FIXES: position = strchr(src, '\0');
}

void good_memchr_1(char *pos, const char *src) {
  pos = strchr(src, '\0');
}

void bad_memchr_2(char *position) {
  position = (char *)memchr("foobar", '\0', 6);
  // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: the length is too short to include the null terminator [bugprone-not-null-terminated-result]
  // CHECK-FIXES: position = strchr("foobar", '\0');
}

void good_memchr_2(char *pos) {
  pos = strchr("foobar", '\0');
}

void bad_memmove(const char *src) {
  char dest[13];
  memmove(dest, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memmove' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest[14];
  // CHECK-FIXES-NEXT: memmove_s(dest, 14, src, strlen(src) + 1);
}

void good_memmove(const char *src) {
  char dst[14];
  memmove_s(dst, 13, src, strlen(src) + 1);
}

void bad_memmove_s(char *dest, const char *src) {
  memmove_s(dest, 13, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memmove_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: memmove_s(dest, 13, src, strlen(src) + 1);
}

void good_memmove_s_1(char *dest, const char *src) {
  memmove_s(dest, 13, src, strlen(src) + 1);
}

void bad_strerror_s(int errno) {
  char dest[13];
  strerror_s(dest, strlen(strerror(errno)), errno);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'strerror_s' is not null-terminated and missing the last character of the error message [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest[14];
  // CHECK-FIXES-NEXT: strerror_s(dest, strlen(strerror(errno)) + 1, errno);
}

void good_strerror_s(int errno) {
  char dst[14];
  strerror_s(dst, strlen(strerror(errno)) + 1, errno);
}

int bad_strncmp_1(char *str0, const char *str1) {
  return strncmp(str0, str1, (strlen(str0) + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: return strncmp(str0, str1, (strlen(str0)));
}

int bad_strncmp_2(char *str2, const char *str3) {
  return strncmp(str2, str3, 1 + strlen(str2));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: return strncmp(str2, str3, strlen(str2));
}

int good_strncmp_1_2(char *str4, const char *str5) {
  return strncmp(str4, str5, strlen(str4));
}

int bad_strncmp_3(char *str6) {
  return strncmp(str6, "string", 7);
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: return strncmp(str6, "string", 6);
}

int good_strncmp_3(char *str7) {
  return strncmp(str7, "string", 6);
}

void bad_strxfrm_1(const char *long_source_name) {
  char long_destination_array_name[13];
  strxfrm(long_destination_array_name, long_source_name,
          strlen(long_source_name));
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: the result from calling 'strxfrm' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char long_destination_array_name[14];
  // CHECK-FIXES-NEXT: strxfrm(long_destination_array_name, long_source_name,
  // CHECK-FIXES-NEXT: strlen(long_source_name) + 1);
}

void good_strxfrm_1(const char *long_source_name) {
  char long_destination_array_name[14];
  strxfrm(long_destination_array_name, long_source_name,
          strlen(long_source_name) + 1);
}

void bad_strxfrm_2() {
  char long_destination_array_name1[16];
  strxfrm(long_destination_array_name1, "long_source_name", 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'strxfrm' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char long_destination_array_name1[17];
  // CHECK-FIXES: strxfrm(long_destination_array_name1, "long_source_name", 17);
}

void good_strxfrm_2() {
  char long_destination_array_name2[17];
  strxfrm(long_destination_array_name2, "long_source_name", 17);
}
