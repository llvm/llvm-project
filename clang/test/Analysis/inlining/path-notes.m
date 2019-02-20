// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.NilArg,osx.cocoa.RetainCount -analyzer-output=text -analyzer-config suppress-null-return-paths=false -fblocks -verify %s
// RUN: %clang_analyze_cc1 -analyzer-checker=core,osx.cocoa.NilArg,osx.cocoa.RetainCount -analyzer-output=plist-multi-file -analyzer-config suppress-null-return-paths=false -fblocks %s -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/path-notes.m.plist -

typedef struct dispatch_queue_s *dispatch_queue_t;
typedef void (^dispatch_block_t)(void);
void dispatch_sync(dispatch_queue_t, dispatch_block_t);

typedef long dispatch_once_t;
// Note: The real dispatch_once has all parameters marked nonnull.
// We don't do that here so that we can trigger a null dereference inside
// the synthesized body.
void dispatch_once(dispatch_once_t *predicate, dispatch_block_t block);


@interface Test
@property int *p;
@end

typedef unsigned long NSUInteger;
typedef signed char BOOL;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
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
@protocol NSFastEnumeration
@end
@protocol NSSecureCoding <NSCoding>
@required
+ (BOOL)supportsSecureCoding;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;
- (id)autorelease;
@end
@interface NSArray : NSObject <NSCopying, NSMutableCopying, NSSecureCoding, NSFastEnumeration>

- (NSUInteger)count;
- (id)objectAtIndex:(NSUInteger)index;

@end

@interface NSArray (NSExtendedArray)
- (NSArray *)arrayByAddingObject:(id)anObject;
- (void)setObject:(id)obj atIndexedSubscript:(NSUInteger)idx __attribute__((availability(macosx,introduced=10.8)));
@end

@interface NSArray (NSArrayCreation)
+ (instancetype)arrayWithObjects:(const id [])objects count:(NSUInteger)cnt;
@end

@interface NSMutableArray : NSArray

- (void)addObject:(id)anObject;
- (void)insertObject:(id)anObject atIndex:(NSUInteger)index;
- (void)removeLastObject;
- (void)removeObjectAtIndex:(NSUInteger)index;
- (void)replaceObjectAtIndex:(NSUInteger)index withObject:(id)anObject;

@end

int *getZeroIfNil(Test *x) {
  return x.p;
  // expected-note@-1 {{'p' not called because the receiver is nil}}
  // expected-note@-2 {{Returning null pointer}}
}

void testReturnZeroIfNil() {
  *getZeroIfNil(0) = 1; // expected-warning{{Dereference of null pointer}}
  // expected-note@-1 {{Calling 'getZeroIfNil'}}
  // expected-note@-2 {{Passing nil object reference via 1st parameter 'x'}}
  // expected-note@-3 {{Returning from 'getZeroIfNil'}}
  // expected-note@-4 {{Dereference of null pointer}}
}


int testDispatchSyncInlining() {
  extern dispatch_queue_t globalQueue;

  __block int x;

  // expected-note@+2 {{Calling 'dispatch_sync'}}
  // expected-note@+1 {{Returning from 'dispatch_sync'}}
  dispatch_sync(globalQueue, ^{
    // expected-note@-1 {{Calling anonymous block}}
    // expected-note@-2 {{Returning to caller}}
    x = 0;
    // expected-note@-1 {{The value 0 is assigned to 'x'}}
  });

  return 1 / x; // expected-warning{{Division by zero}}
  // expected-note@-1 {{Division by zero}}
}

int testDispatchSyncInliningNoPruning(int coin) {
  // This tests exactly the same case as above, except on a bug report where
  // path pruning is disabled (an uninitialized variable capture).
  // In this case 
  extern dispatch_queue_t globalQueue;

  __block int y;

  // expected-note@+1 {{Calling 'dispatch_sync'}}
  dispatch_sync(globalQueue, ^{
    // expected-note@-1 {{Calling anonymous block}}
    int x;
    // expected-note@-1 {{'x' declared without an initial value}}
    ^{ y = x; }(); // expected-warning{{Variable 'x' is uninitialized when captured by block}}
    // expected-note@-1 {{'x' is uninitialized when captured by block}}
  });

  return y;
}


@interface PointerWrapper
- (int *)getPtr;
@end

id getNil() {
  return 0;
}

void testNilReceiverHelper(int *x) {
  *x = 1; // expected-warning {{Dereference of null pointer}}
  // expected-note@-1 {{Dereference of null pointer (loaded from variable 'x')}}
}

void testNilReceiver(id *x, id *y, id *z) {
  // FIXME: Should say "Assuming pointer value is null" instead.
  // For some reason we're displaying different notes for
  // tracked and untracked pointers.
  if (*y) {} // expected-note    {{Assuming the condition is false}}
             // expected-note@-1 {{Taking false branch}}
  if (*x) { // expected-note {{Assuming pointer value is null}}
    // expected-note@-1 {{Taking false branch}}
    return;
  }
  // FIXME: Should say "Assuming pointer value is null" instead.
  if (*z) {} // expected-note    {{Assuming the condition is false}}
             // expected-note@-1 {{Taking false branch}}
  testNilReceiverHelper([*x getPtr]);
  // expected-note@-1 {{'getPtr' not called because the receiver is nil}}
  // expected-note@-2 {{Passing null pointer value via 1st parameter 'x'}}
  // expected-note@-3 {{Calling 'testNilReceiverHelper'}}
}

id testCreateArrayLiteral(id myNil) {
  if (myNil) // expected-note {{Assuming 'myNil' is nil}}
    ;        // expected-note@-1 {{Taking false branch}}
  return @[ @"a", myNil, @"c" ]; // expected-warning {{Array element cannot be nil}}
                                 //expected-note@-1 {{Array element cannot be nil}}
}

// <rdar://problem/14611722>
id testAutoreleaseTakesEffectInDispatch() {
  static dispatch_once_t token = 0;
  dispatch_once(&token, ^{});

  id x = [[[[NSObject alloc] init] autorelease] autorelease];
  // expected-note@-1 {{Method returns an instance of NSObject with a +1 retain count}}
  // expected-note@-2 {{Object autoreleased}}
  // expected-note@-3 {{Object autoreleased}}

  dispatch_once(&token, ^{}); // don't crash, don't warn here

  return x; // expected-warning{{Object autoreleased too many times}}
  // expected-note@-1 {{Object was autoreleased 2 times but the object has a +0 retain count}}
}

void testNullDereferenceInDispatch() {
  dispatch_once(0, ^{}); // no-warning, don't crash
}

