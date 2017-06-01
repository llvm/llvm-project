// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -analyzer-checker=deadcode.DeadStores,osx.cocoa.RetainCount -fblocks -verify -Wno-objc-root-class %s
// expected-no-diagnostics

typedef signed char BOOL;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSValue : NSObject <NSCopying, NSCoding>  - (void)getValue:(void *)value; @end
typedef float CGFloat;
typedef struct _NSPoint {} NSRange;
@interface NSValue (NSValueRangeExtensions)  + (NSValue *)valueWithRange:(NSRange)range;
- (BOOL)containsObject:(id)anObject;
@end
@class NSURLAuthenticationChallenge;
@interface NSResponder : NSObject <NSCoding> {} @end
@class NSArray, NSDictionary, NSString;
@interface NSObject (NSKeyValueBindingCreation)
+ (void)exposeBinding:(NSString *)binding;
- (NSArray *)exposedBindings;
@end
extern NSString *NSAlignmentBinding;

// This test case was reported as a false positive due to a bug in the
// LiveVariables <-> deadcode.DeadStores interplay.  We should not flag a warning
// here.  The test case was reported in:
//  http://lists.llvm.org/pipermail/cfe-dev/2008-July/002157.html
void DeadStoreTest(NSObject *anObject) {
  NSArray *keys;
  if ((keys = [anObject exposedBindings]) &&   // no-warning
      ([keys containsObject:@"name"] && [keys containsObject:@"icon"])) {}
}

// This test case was a false positive due to how clang models
// pointer types and ObjC object pointer types differently.  Here
// we don't warn about a dead store because 'nil' is assigned to
// an object pointer for the sake of defensive programming.
void rdar_7631278(NSObject *x) {
  x = ((void*)0);
}

// This test case issuing a bogus warning for the declaration of 'isExec'
// because the compound statement for the @synchronized was being visited
// twice by the LiveVariables analysis.
BOOL baz_rdar8527823();
void foo_rdar8527823();
@interface RDar8527823
- (void) bar_rbar8527823;
@end
@implementation RDar8527823
- (void) bar_rbar8527823
{
 @synchronized(self) {
   BOOL isExec = baz_rdar8527823(); // no-warning
   if (isExec) foo_rdar8527823();
 }
}
@end

// Don't flag dead stores to assignments to self within a nested assignment.
@interface Rdar7947686
- (id) init;
@end

@interface Rdar7947686_B : Rdar7947686
- (id) init;
@end

@implementation Rdar7947686_B
- (id) init {
  id x = (self = [super init]); // no-warning
  return x;
}
@end

// Don't flag dead stores when a variable is captured in a block used
// by a property access.
@interface RDar10591355
@property (assign) int x;
@end

RDar10591355 *rdar10591355_aux();

void rdar10591355() {
  RDar10591355 *p = rdar10591355_aux();
  ^{ (void) p.x; }();
}

@interface Radar11059352_1 {
@private
    int *_pathString;
}
@property int *pathString;
@end
@interface Radar11059352 {
@private
Radar11059352_1 *_Path;
}
@end
@implementation Radar11059352

- (int*)usePath {
    Radar11059352_1 *xxxxx = _Path; // no warning
    int *wp = xxxxx.pathString;
    return wp;
}
@end

id test_objc_precise_lifetime_foo();
void test_objc_precise_lifetime() {
  __attribute__((objc_precise_lifetime)) id dead = test_objc_precise_lifetime_foo(); // no-warning
  dead = 0;
  dead = test_objc_precise_lifetime_foo(); // no-warning
  dead = 0;
}
