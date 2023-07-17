// RUN: %clang_cc1  -fsyntax-only -verify -Weverything %s
// expected-no-diagnostics
// rdar://12103434

@class NSString;

@interface NSObject @end

@interface MyClass  : NSObject

@property (nonatomic, copy, readonly) NSString* name;

@end

@interface MyClass () {
    NSString* _name;
}

@property (nonatomic, copy) NSString* name;

@end

@implementation MyClass

@synthesize name = _name;

@end
