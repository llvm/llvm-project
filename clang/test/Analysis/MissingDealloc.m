// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.osx.cocoa.Dealloc -fblocks %s 2>&1 | FileCheck -check-prefix=CHECK %s
// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.osx.cocoa.Dealloc -fblocks -triple x86_64-apple-darwin10 -fobjc-arc %s 2>&1 | FileCheck -check-prefix=CHECK-ARC -allow-empty '--implicit-check-not=error:' '--implicit-check-not=warning:' %s

typedef signed char BOOL;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (Class)class;
@end

@interface NSObject <NSObject> {}
- (void)dealloc;
- (id)init;
@end

typedef struct objc_selector *SEL;

//===------------------------------------------------------------------------===
// Do not warn about missing -dealloc method.  Not enough context to know
// whether the ivar is retained or not.

@interface MissingDeallocWithIvar : NSObject {
  NSObject *_ivar;
}
@end

@implementation MissingDeallocWithIvar
@end

//===------------------------------------------------------------------------===
// Do not warn about missing -dealloc method.  These properties are not
// retained or synthesized.

@interface MissingDeallocWithIntProperty : NSObject
@property (assign) int ivar;
@end

@implementation MissingDeallocWithIntProperty
@end

@interface MissingDeallocWithSELProperty : NSObject
@property (assign) SEL ivar;
@end

@implementation MissingDeallocWithSELProperty
@end

//===------------------------------------------------------------------------===
// Warn about missing -dealloc method.

@interface MissingDeallocWithCopyProperty : NSObject
@property (copy) NSObject *ivar;
@end

// CHECK: MissingDealloc.m:[[@LINE+1]]:1: warning: Objective-C class 'MissingDeallocWithCopyProperty' lacks a 'dealloc' instance method
@implementation MissingDeallocWithCopyProperty
@end

@interface MissingDeallocWithRetainProperty : NSObject
@property (retain) NSObject *ivar;
@end

// CHECK: MissingDealloc.m:[[@LINE+1]]:1: warning: Objective-C class 'MissingDeallocWithRetainProperty' lacks a 'dealloc' instance method
@implementation MissingDeallocWithRetainProperty
@end

@interface MissingDeallocWithIVarAndRetainProperty : NSObject {
  NSObject *_ivar2;
}
@property (retain) NSObject *ivar1;
@end

// CHECK: MissingDealloc.m:[[@LINE+1]]:1: warning: Objective-C class 'MissingDeallocWithIVarAndRetainProperty' lacks a 'dealloc' instance method
@implementation MissingDeallocWithIVarAndRetainProperty
@end

@interface MissingDeallocWithReadOnlyRetainedProperty : NSObject
@property (readonly,retain) NSObject *ivar;
@end

// CHECK: MissingDealloc.m:[[@LINE+1]]:1: warning: Objective-C class 'MissingDeallocWithReadOnlyRetainedProperty' lacks a 'dealloc' instance method
@implementation MissingDeallocWithReadOnlyRetainedProperty
@end


//===------------------------------------------------------------------------===
//  Don't warn about iVars that are selectors.

@interface TestSELs : NSObject {
  SEL a;
  SEL b;
}

@end

@implementation TestSELs
- (id)init {
  if( (self = [super init]) ) {
    a = @selector(a);
    b = @selector(b);
  }

  return self;
}
@end

//===------------------------------------------------------------------------===
//  Don't warn about iVars that are IBOutlets.

@class NSWindow;

@interface HasOutlet : NSObject {
IBOutlet NSWindow *window;
}
@end

@implementation HasOutlet // no-warning
@end

//===------------------------------------------------------------------------===
// PR 3187: http://llvm.org/bugs/show_bug.cgi?id=3187
// - Disable the missing -dealloc check for classes that subclass SenTestCase

@class NSString;

@interface SenTestCase : NSObject {}
@end

@interface MyClassTest : SenTestCase {
  NSString *resourcePath;
}
@end

@interface NSBundle : NSObject {}
+ (NSBundle *)bundleForClass:(Class)aClass;
- (NSString *)resourcePath;
@end

@implementation MyClassTest
- (void)setUp {
  resourcePath = [[NSBundle bundleForClass:[self class]] resourcePath];
}
- (void)testXXX {
  // do something which uses resourcepath
}
@end
// CHECK: 4 warnings generated.
