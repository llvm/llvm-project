// RUN: %check_clang_tidy %s bugprone-not-null-terminated-result %t -- \
// RUN: -- -std=c++11 -I %S/Inputs/not-null-terminated-result

// FIXME: Something wrong with the APInt un/signed conversion on Windows:
// in 'wcsncmp(wcs6, L"string", 7);' it tries to inject '4294967302' as length.

// UNSUPPORTED: system-windows

#include "not-null-terminated-result-cxx.h"

#define __STDC_LIB_EXT1__ 1
#define __STDC_WANT_LIB_EXT1__ 1

void bad_wmemchr_1(wchar_t *position, const wchar_t *src) {
  position = (wchar_t *)wmemchr(src, L'\0', wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:45: warning: the length is too short to include the null terminator [bugprone-not-null-terminated-result]
  // CHECK-FIXES: position = wcschr(src, L'\0');
}

void good_wmemchr_1(wchar_t *pos, const wchar_t *src) {
  pos = wcschr(src, L'\0');
}

void bad_wmemchr_2(wchar_t *position) {
  position = (wchar_t *)wmemchr(L"foobar", L'\0', 6);
  // CHECK-MESSAGES: :[[@LINE-1]]:51: warning: the length is too short to include the null terminator [bugprone-not-null-terminated-result]
  // CHECK-FIXES: position = wcschr(L"foobar", L'\0');
}

void good_wmemchr_2(wchar_t *pos) {
  pos = wcschr(L"foobar", L'\0');
}


void bad_wmemmove(const wchar_t *src) {
  wchar_t dest[13];
  wmemmove(dest, src, wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemmove' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t dest[14];
  // CHECK-FIXES-NEXT: wmemmove_s(dest, 14, src, wcslen(src) + 1);
}

void good_wmemmove(const wchar_t *src) {
  wchar_t dst[14];
  wmemmove_s(dst, 13, src, wcslen(src) + 1);
}

void bad_wmemmove_s(wchar_t *dest, const wchar_t *src) {
  wmemmove_s(dest, 13, src, wcslen(src));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wmemmove_s' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wmemmove_s(dest, 13, src, wcslen(src) + 1);
}

void good_wmemmove_s_1(wchar_t *dest, const wchar_t *src) {
  wmemmove_s(dest, 13, src, wcslen(src) + 1);
}

int bad_wcsncmp_1(wchar_t *wcs0, const wchar_t *wcs1) {
  return wcsncmp(wcs0, wcs1, (wcslen(wcs0) + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wcsncmp(wcs0, wcs1, (wcslen(wcs0)));
}

int bad_wcsncmp_2(wchar_t *wcs2, const wchar_t *wcs3) {
  return wcsncmp(wcs2, wcs3, 1 + wcslen(wcs2));
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wcsncmp(wcs2, wcs3, wcslen(wcs2));
}

int good_wcsncmp_1_2(wchar_t *wcs4, const wchar_t *wcs5) {
  return wcsncmp(wcs4, wcs5, wcslen(wcs4));
}

int bad_wcsncmp_3(wchar_t *wcs6) {
  return wcsncmp(wcs6, L"string", 7);
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: comparison length is too long and might lead to a buffer overflow [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wcsncmp(wcs6, L"string", 6);
}

int good_wcsncmp_3(wchar_t *wcs7) {
  return wcsncmp(wcs7, L"string", 6);
}

void bad_wcsxfrm_1(const wchar_t *long_source_name) {
  wchar_t long_destination_array_name[13];
  wcsxfrm(long_destination_array_name, long_source_name,
          wcslen(long_source_name));
  // CHECK-MESSAGES: :[[@LINE-2]]:3: warning: the result from calling 'wcsxfrm' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t long_destination_array_name[14];
  // CHECK-FIXES-NEXT: wcsxfrm(long_destination_array_name, long_source_name,
  // CHECK-FIXES-NEXT: wcslen(long_source_name) + 1);
}

void good_wcsxfrm_1(const wchar_t *long_source_name) {
  wchar_t long_destination_array_name[14];
  wcsxfrm(long_destination_array_name, long_source_name,
          wcslen(long_source_name) + 1);
}

void bad_wcsxfrm_2() {
  wchar_t long_destination_array_name1[16];
  wcsxfrm(long_destination_array_name1, L"long_source_name", 16);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: the result from calling 'wcsxfrm' is not null-terminated [bugprone-not-null-terminated-result]
  // CHECK-FIXES: wchar_t long_destination_array_name1[17];
  // CHECK-FIXES: wcsxfrm(long_destination_array_name1, L"long_source_name", 17);
}

void good_wcsxfrm_2() {
  wchar_t long_destination_array_name2[17];
  wcsxfrm(long_destination_array_name2, L"long_source_name", 17);
}
