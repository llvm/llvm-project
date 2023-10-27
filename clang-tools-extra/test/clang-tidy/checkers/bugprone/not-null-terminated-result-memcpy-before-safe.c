// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -config="{CheckOptions: \
// RUN: {bugprone-not-null-terminated-result.WantToUseSafeFunction: true}}" \
// RUN: -- -std=c11 -I %S/Inputs/not-null-terminated-result

#include "not-null-terminated-result-c.h"

// The following is not defined therefore the safe functions are unavailable.
// #define __STDC_LIB_EXT1__ 1

#define __STDC_WANT_LIB_EXT1__ 1

//===----------------------------------------------------------------------===//
// memcpy() - destination array tests
//===----------------------------------------------------------------------===//

void bad_memcpy_not_just_char_dest(const char *src) {
  unsigned char dest00[13];
  memcpy(dest00, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: unsigned char dest00[14];
  // CHECK-FIXES-NEXT: strcpy((char *)dest00, src);
}

void good_memcpy_not_just_char_dest(const char *src) {
  unsigned char dst00[14];
  strcpy((char *)dst00, src);
}

void bad_memcpy_known_dest(const char *src) {
  char dest01[13];
  memcpy(dest01, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strcpy(dest01, src);
}

void good_memcpy_known_dest(const char *src) {
  char dst01[13];
  strcpy(dst01, src);
}

//===----------------------------------------------------------------------===//
// memcpy() - length tests
//===----------------------------------------------------------------------===//

void bad_memcpy_full_source_length(const char *src) {
  char dest20[13];
  memcpy(dest20, src, strlen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strcpy(dest20, src);
}

void good_memcpy_full_source_length(const char *src) {
  char dst20[13];
  strcpy(dst20, src);
}

void bad_memcpy_partial_source_length(const char *src) {
  char dest21[13];
  memcpy(dest21, src, strlen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'memcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: strncpy(dest21, src, strlen(src) - 1);
  // CHECK-FIXES-NEXT: dest21[strlen(src) - 1] = '\0';
}

void good_memcpy_partial_source_length(const char *src) {
  char dst21[13];
  strncpy(dst21, src, strlen(src) - 1);
  dst21[strlen(src) - 1] = '\0';
}
