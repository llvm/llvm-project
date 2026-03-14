// RUN: %clang_analyze_cc1 -triple thumbv7-apple-ios11.0 -verify=available \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s
// RUN: %clang_analyze_cc1 -triple thumbv7-apple-ios10.0 -verify=notavailable \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-macos10.13 -verify=available \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-macos10.12 -verify=notavailable \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s
// RUN: %clang_analyze_cc1 -triple thumbv7-apple-watchos4.0 -verify=available \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s
// RUN: %clang_analyze_cc1 -triple thumbv7-apple-watchos3.0 -verify=notavailable \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s
// RUN: %clang_analyze_cc1 -triple thumbv7-apple-tvos11.0 -verify=available \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s
// RUN: %clang_analyze_cc1 -triple thumbv7-apple-tvos10.0 -verify=notavailable \
// RUN:     -analyzer-checker=security.insecureAPI.decodeValueOfObjCType %s

// notavailable-no-diagnostics

typedef unsigned long NSUInteger;

@interface NSCoder
- (void)decodeValueOfObjCType:(const char *)type
                           at:(void *)data;
- (void)decodeValueOfObjCType:(const char *)type
                           at:(void *)data
                         size:(NSUInteger)size;
@end

void test(NSCoder *decoder) {
  // This would be a vulnerability on 64-bit platforms
  // but not on 32-bit platforms.
  NSUInteger x;
  [decoder decodeValueOfObjCType:"I" at:&x]; // available-warning{{Deprecated method '-decodeValueOfObjCType:at:' is insecure as it can lead to potential buffer overflows. Use the safer '-decodeValueOfObjCType:at:size:' method}}
  [decoder decodeValueOfObjCType:"I" at:&x size:sizeof(x)]; // no-warning
}
