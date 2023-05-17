// RUN: %check_clang_tidy %s objc-assert-equals %t -- -- -I %S/Inputs/assert
#include "XCTestAssertions.h"
// Can't reference NSString directly so we use this getStr() instead.
__typeof(@"abc") getStr(void) {
  return @"abc";
}
void foo(void) {
  XCTAssertEqual(getStr(), @"abc");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use XCTAssertEqualObjects for comparing objects
  // CHECK-FIXES: XCTAssertEqualObjects(getStr(), @"abc");
  XCTAssertEqual(@"abc", @"abc");
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use XCTAssertEqualObjects for comparing objects
  // CHECK-FIXES: XCTAssertEqualObjects(@"abc", @"abc");
  XCTAssertEqual(@"abc", getStr());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use XCTAssertEqualObjects for comparing objects
  // CHECK-FIXES: XCTAssertEqualObjects(@"abc", getStr());
  XCTAssertEqual(getStr(), getStr());
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: use XCTAssertEqualObjects for comparing objects
  // CHECK-FIXES: XCTAssertEqualObjects(getStr(), getStr());
  // Primitive types should be ok
  XCTAssertEqual(123, 123);
  XCTAssertEqual(123.0, 123.45);
  // FIXME: This is the case where we don't diagnose properly.
  // XCTAssertEqual(@"abc" != @"abc", @"xyz" != @"xyz")
}
