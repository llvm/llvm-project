// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -std=c++11 -I %S/Inputs/not-null-terminated-result

#include "not-null-terminated-result-cxx.h"

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1

//===----------------------------------------------------------------------===//
// wmemcpy() - destination array tests
//===----------------------------------------------------------------------===//

void bad_wmemcpy_known_dest(const wchar_t *src) {
  wchar_t dest01[13];
  wmemcpy(dest01, src, wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t dest01[14];
  // CHECK-FIXES-NEXT: wcscpy_s(dest01, src);
}

void good_wmemcpy_known_dest(const wchar_t *src) {
  wchar_t dst01[14];
  wcscpy_s(dst01, src);
}

//===----------------------------------------------------------------------===//
// wmemcpy() - length tests
//===----------------------------------------------------------------------===//

void bad_wmemcpy_full_source_length(const wchar_t *src) {
  wchar_t dest20[13];
  wmemcpy(dest20, src, wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t dest20[14];
  // CHECK-FIXES-NEXT: wcscpy_s(dest20, src);
}

void good_wmemcpy_full_source_length(const wchar_t *src) {
  wchar_t dst20[14];
  wcscpy_s(dst20, src);
}

void bad_wmemcpy_partial_source_length(const wchar_t *src) {
  wchar_t dest21[13];
  wmemcpy(dest21, src, wcslen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemcpy' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t dest21[14];
  // CHECK-FIXES-NEXT: wcsncpy_s(dest21, src, wcslen(src) - 1);
}

void good_wmemcpy_partial_source_length(const wchar_t *src) {
  wchar_t dst21[14];
  wcsncpy_s(dst21, src, wcslen(src) - 1);
}

//===----------------------------------------------------------------------===//
// wmemcpy_s() - destination array tests
//===----------------------------------------------------------------------===//

void bad_wmemcpy_s_unknown_dest(wchar_t *dest40, const wchar_t *src) {
  wmemcpy_s(dest40, 13, src, wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wcscpy_s(dest40, 13, src);
}

void good_wmemcpy_s_unknown_dest(wchar_t *dst40, const wchar_t *src) {
  wcscpy_s(dst40, 13, src);
}

void bad_wmemcpy_s_known_dest(const wchar_t *src) {
  wchar_t dest41[13];
  wmemcpy_s(dest41, 13, src, wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t dest41[14];
  // CHECK-FIXES-NEXT: wcscpy_s(dest41, src);
}

void good_wmemcpy_s_known_dest(const wchar_t *src) {
  wchar_t dst41[13];
  wcscpy_s(dst41, src);
}

//===----------------------------------------------------------------------===//
// wmemcpy_s() - length tests
//===----------------------------------------------------------------------===//

void bad_wmemcpy_s_full_source_length(const wchar_t *src) {
  wchar_t dest60[13];
  wmemcpy_s(dest60, 13, src, wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t dest60[14];
  // CHECK-FIXES-NEXT: wcscpy_s(dest60, src);
}

void good_wmemcpy_s_full_source_length(const wchar_t *src) {
  wchar_t dst60[13];
  wcscpy_s(dst60, src);
}

void bad_wmemcpy_s_partial_source_length(const wchar_t *src) {
  wchar_t dest61[13];
  wmemcpy_s(dest61, 13, src, wcslen(src) - 1);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemcpy_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t dest61[14];
  // CHECK-FIXES-NEXT: wcsncpy_s(dest61, src, wcslen(src) - 1);
}

void good_wmemcpy_s_partial_source_length(const wchar_t *src) {
  wchar_t dst61[13];
  wcsncpy_s(dst61, src, wcslen(src) - 1);
}
