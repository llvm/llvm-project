// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.NoUnretainedMemberChecker \
// RUN:   -analyzer-output=plist -o %t.plist %s
// RUN: FileCheck --input-file=%t.plist %s

#include "objc-mock-types.h"

@interface Writer : NSObject
@property (nonatomic, copy) NSString *name;
@end

@implementation Writer
- (void)first {
  _name = nil;
}

- (void)second {
  _name = nil;
}

static void writeFromFunction(Writer *other) {
  other->_name = nil;
}
@end

// CHECK:      <key>issue_context_kind</key><string>Objective-C method</string>
// CHECK-NEXT: <key>issue_context</key><string>first</string>
// CHECK:      <key>issue_context_kind</key><string>Objective-C method</string>
// CHECK-NEXT: <key>issue_context</key><string>second</string>
// CHECK:      <key>issue_context_kind</key><string>function</string>
// CHECK-NEXT: <key>issue_context</key><string>writeFromFunction</string>
