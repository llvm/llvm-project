// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -objcmt-migrate-ns-macros -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result

typedef long NSInteger;
enum {
    UIViewNone         = 0x0,
    UIViewMargin       = 0x1,
    UIViewWidth        = 0x2,
    UIViewRightMargin  = 0x3,
    UIViewBottomMargin = 0xbadbeef
};
typedef NSInteger UITableStyle;


typedef
  enum { two = 1 } NumericEnum2;

typedef enum { three = 1 } NumericEnum3;

typedef enum { four = 1 } NumericEnum4;

