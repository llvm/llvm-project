// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -Wno-objc-root-class \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-config core.CallAndMessage:FunctionPointer=false \
// RUN:   -analyzer-config core.CallAndMessage:ParameterCount=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXThisMethodCall=false \
// RUN:   -analyzer-config core.CallAndMessage:CXXDeallocationArg=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:ArgPointeeInitializedness=false \
// RUN:   -analyzer-config core.CallAndMessage:NilReceiver=false \
// RUN:   -analyzer-config core.CallAndMessage:UndefReceiver=true \
// RUN:   -analyzer-output=plist -o %t.plist
// RUN: cat %t.plist | FileCheck %s

//===----------------------------------------------------------------------===//
// The following code is reduced using delta-debugging from
// Foundation.h (Mac OS X).
//
// It includes the basic definitions for the test cases below.
// Not directly including Foundation.h directly makes this test case
// both svelte and portable to non-Mac platforms.
//===----------------------------------------------------------------------===//

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying
- (id)mutableCopyWithZone:(NSZone *)zone;
@end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {
}
@end
@class NSString, NSData;
@class NSString, NSData, NSMutableData, NSMutableDictionary, NSMutableArray;
typedef struct {
} NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id *)stackbuf count:(NSUInteger)len;
@end
@class NSData, NSIndexSet, NSString, NSURL;
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSCoding, NSFastEnumeration>
- (NSUInteger)count;
@end
@interface NSArray (NSArrayCreation)
+ (id)array;
- (NSUInteger)length;
- (void)addObject:(id)object;
@end
extern NSString *const NSUndoManagerCheckpointNotification;

//===----------------------------------------------------------------------===//
// Test cases.
//===----------------------------------------------------------------------===//

unsigned f1() {
  NSString *aString;
  return [aString length]; // expected-warning {{Receiver in message expression is an uninitialized value [core.CallAndMessage]}}
}

// TODO: If this hash ever changes, turn core.CallAndMessage:UndefReceiver from
// a checker option into a checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>29873175e1cc0a98f7040057279925a0</string>

@interface RDar9241180
@property(readwrite, assign) id x;
- (id)testAnalyzer1:(int)y;
@end

@implementation RDar9241180
@synthesize x;
- (id)testAnalyzer1:(int)y {
  RDar9241180 *o;
  if (y && o.x) // expected-warning {{Property access on an uninitialized object pointer [core.CallAndMessage]}}
    return o;

  // TODO: If this hash ever changes, turn core.CallAndMessage:UndefReceiver from
  // a checker option into a checker, as described in the CallAndMessage comments!
  // CHECK: <key>issue_hash_content_of_line_in_context</key>
  // CHECK-SAME: <string>00ddd30796a283de33e662da8449c796</string>

  return o; // expected-warning {{Undefined or garbage value returned to caller [core.uninitialized.UndefReturn]}}
}
@end

// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>8d468e24df7d887f4182bf49f5dd8b71</string>

typedef signed char BOOL;
typedef unsigned int NSUInteger;

@interface Subscriptable : NSObject
- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)index;
- (id)objectAtIndexedSubscript:(NSUInteger)index;

- (void)setObject:(id)obj forKeyedSubscript:(id)key;
- (id)objectForKeyedSubscript:(id)key;
@end

@interface Test : Subscriptable
@end

@implementation Test

// <rdar://problem/9241180> for subscripting
- (id)testUninitializedObject:(BOOL)keyed {
  Test *o;
  if (keyed) {
    if (o[self]) // expected-warning {{Subscript access on an uninitialized object pointer [core.CallAndMessage]}}
      return o;  // no-warning (sink)
  } else {
    if (o[0])   // expected-warning {{Subscript access on an uninitialized object pointer [core.CallAndMessage]}}
      return o; // no-warning (sink)
  }
  return self;
}
@end

// TODO: If this hash ever changes, turn core.CallAndMessage:UndefReceiver from
// a checker option into a checker, as described in the CallAndMessage comments!
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>8d943563d78377fc5dfcd4fdde904e5e</string>
// CHECK: <key>issue_hash_content_of_line_in_context</key>
// CHECK-SAME: <string>9a2a9698763d62bed38d91fe5fb4aefd</string>
