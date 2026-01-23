// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -I %S/Inputs/not-null-terminated-result

#include "not-null-terminated-result-c.h"

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1

//===----------------------------------------------------------------------===//
// memcpy() - destination array tests
//===----------------------------------------------------------------------===//

void bad_memcpy_not_just_char_dest(const char *src) {
  unsigned char dest00[13];
  memcpy(dest00, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: unsigned char dest00[14];
  // CHECK-FIXES-NEXT: strcpy_s((char *)dest00, 14, src);
}

void good_memcpy_not_just_char_dest(const char *src) {
  unsigned char dst00[14];
  strcpy_s((char *)dst00, 14, src);
}

void bad_memcpy_known_dest(const char *src) {
  char dest01[13];
  memcpy(dest01, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest01[14];
  // CHECK-FIXES: strcpy_s(dest01, 14, src);
}

void good_memcpy_known_dest(const char *src) {
  char dst01[14];
  strcpy_s(dst01, 14, src);
}

//===----------------------------------------------------------------------===//
// memcpy() - length tests
//===----------------------------------------------------------------------===//

void bad_memcpy_full_source_length(const char *src) {
  char dest20[13];
  memcpy(dest20, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest20[14];
  // CHECK-FIXES-NEXT: strcpy_s(dest20, 14, src);
}

void good_memcpy_full_source_length(const char *src) {
  char dst20[14];
  strcpy_s(dst20, 14, src);
}

void bad_memcpy_partial_source_length(const char *src) {
  char dest21[13];
  memcpy(dest21, src, strlen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest21[14];
  // CHECK-FIXES-NEXT: strncpy_s(dest21, 14, src, strlen(src) - 1);
}

void good__memcpy_partial_source_length(const char *src) {
  char dst21[14];
  strncpy_s(dst21, 14, src, strlen(src) - 1);
}

//===----------------------------------------------------------------------===//
// memcpy_s() - destination array tests
//===----------------------------------------------------------------------===//

void bad_memcpy_s_unknown_dest(char *dest40, const char *src) {
  memcpy_s(dest40, 13, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strcpy_s(dest40, 13, src);
}

void good_memcpy_s_unknown_dest(char *dst40, const char *src) {
  strcpy_s(dst40, 13, src);
}

void bad_memcpy_s_known_dest(const char *src) {
  char dest41[13];
  memcpy_s(dest41, 13, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest41[14];
  // CHECK-FIXES-NEXT: strcpy_s(dest41, 14, src);
}

void good_memcpy_s_known_dest(const char *src) {
  char dst41[14];
  strcpy_s(dst41, 14, src);
}

//===----------------------------------------------------------------------===//
// memcpy_s() - length tests
//===----------------------------------------------------------------------===//

void bad_memcpy_s_full_source_length(const char *src) {
  char dest60[13];
  memcpy_s(dest60, 13, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest60[14];
  // CHECK-FIXES-NEXT: strcpy_s(dest60, 14, src);
}

void good_memcpy_s_full_source_length(const char *src) {
  char dst60[14];
  strcpy_s(dst60, 14, src);
}

void bad_memcpy_s_partial_source_length(const char *src) {
  char dest61[13];
  memcpy_s(dest61, 13, src, strlen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: char dest61[14];
  // CHECK-FIXES-NEXT: strncpy_s(dest61, 14, src, strlen(src) - 1);
}

void good_memcpy_s_partial_source_length(const char *src) {
  char dst61[14];
  strncpy_s(dst61, 14, src, strlen(src) - 1);
}
