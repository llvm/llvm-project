// RUN: cp %s %t
// RUN: %clang_cc1 -x objective-c -triple x86_64-apple-darwin -Wno-objc-root-class -pedantic -Wall -fixit %t
// RUN: %clang_cc1 -x objective-c -triple x86_64-apple-darwin -Wno-objc-root-class -fsyntax-only -pedantic -Wall -Werror %t
// RUN: %clang_cc1 -x objective-c -triple x86_64-apple-darwin -Wno-objc-root-class -E -o - %t | FileCheck %s

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>  - (NSUInteger)length; @end
extern void NSLog(NSString *format, ...);

/* This is a test of the various code modification hints that are
   provided as part of warning or extension diagnostics. All of the
   warnings will be fixed by -fixit, and the resulting file should
   compile cleanly with -Werror -pedantic. */

extern NSString *NonliteralString;

void test() {
  // nonliteral format
  NSLog(NonliteralString);
}

// Validate the fixes.
// CHECK: NSLog(@"%@", NonliteralString);
