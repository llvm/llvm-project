// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.Dealloc -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=osx.cocoa.Dealloc -fblocks -verify -triple x86_64-apple-darwin10 -fobjc-arc %s

#define NON_ARC !__has_feature(objc_arc)

// No diagnostics expected under ARC.
#if !NON_ARC
  // expected-no-diagnostics
#endif

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

#if NON_ARC
// expected-warning@+2{{'MissingDeallocWithCopyProperty' lacks a 'dealloc' instance method but must release '_ivar'}}
#endif
@implementation MissingDeallocWithCopyProperty
@end

@interface MissingDeallocWithRetainProperty : NSObject
@property (retain) NSObject *ivar;
@end

#if NON_ARC
// expected-warning@+2{{'MissingDeallocWithRetainProperty' lacks a 'dealloc' instance method but must release '_ivar'}}
#endif
@implementation MissingDeallocWithRetainProperty
@end

@interface MissingDeallocWithMultipleProperties : NSObject
@property (retain) NSObject *ivar1;
@property (retain) NSObject *ivar2;
@end

#if NON_ARC
// expected-warning@+2{{'MissingDeallocWithMultipleProperties' lacks a 'dealloc' instance method but must release '_ivar1' and others}}
#endif
@implementation MissingDeallocWithMultipleProperties
@end

@interface MissingDeallocWithIVarAndRetainProperty : NSObject {
  NSObject *_ivar2;
}
@property (retain) NSObject *ivar1;
@end

#if NON_ARC
// expected-warning@+2{{'MissingDeallocWithIVarAndRetainProperty' lacks a 'dealloc' instance method but must release '_ivar1'}}
#endif
@implementation MissingDeallocWithIVarAndRetainProperty
@end

@interface MissingDeallocWithReadOnlyRetainedProperty : NSObject
@property (readonly,retain) NSObject *ivar;
@end

#if NON_ARC
// expected-warning@+2{{'MissingDeallocWithReadOnlyRetainedProperty' lacks a 'dealloc' instance method but must release '_ivar'}}
#endif
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

@property (retain) NSObject *ivar;

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

//===------------------------------------------------------------------------===
// Don't warn for clases that aren't subclasses of NSObject

__attribute__((objc_root_class))
@interface NonNSObjectMissingDealloc
@property (retain) NSObject *ivar;
@end
@implementation NonNSObjectMissingDealloc
@end

// CHECK: 4 warnings generated.
