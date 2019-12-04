// RUN: %clang_analyze_cc1  -Wno-objc-literal-conversion -analyzer-checker=core,osx.cocoa.NonNilReturnValue,osx.cocoa.NilArg,osx.cocoa.Loops,debug.ExprInspection -verify -Wno-objc-root-class %s

void clang_analyzer_eval(int);

#define nil ((id)0)

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
@protocol NSSecureCoding <NSCoding>
@required
+ (BOOL)supportsSecureCoding;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)alloc;

- (id)mutableCopy;
@end

typedef struct {
  unsigned long state;
  id *itemsPtr;
  unsigned long *mutationsPtr;
  unsigned long extra[5];
} NSFastEnumerationState;
@protocol NSFastEnumeration
- (NSUInteger)countByEnumeratingWithState:(NSFastEnumerationState *)state objects:(id [])buffer count:(NSUInteger)len;
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

@interface NSDictionary : NSObject <NSCopying, NSMutableCopying, NSSecureCoding, NSFastEnumeration>

- (NSUInteger)count;
- (id)objectForKey:(id)aKey;
- (NSEnumerator *)keyEnumerator;

@end

@interface NSDictionary (NSDictionaryCreation)

+ (id)dictionary;
+ (id)dictionaryWithObject:(id)object forKey:(id <NSCopying>)key;
+ (instancetype)dictionaryWithObjects:(const id [])objects forKeys:(const id <NSCopying> [])keys count:(NSUInteger)cnt;

@end

@interface NSMutableDictionary : NSDictionary

- (void)removeObjectForKey:(id)aKey;
- (void)setObject:(id)anObject forKey:(id <NSCopying>)aKey;

@end

@interface NSMutableDictionary (NSExtendedMutableDictionary)

- (void)addEntriesFromDictionary:(NSDictionary *)otherDictionary;
- (void)removeAllObjects;
- (void)removeObjectsForKeys:(NSArray *)keyArray;
- (void)setDictionary:(NSDictionary *)otherDictionary;
- (void)setObject:(id)obj forKeyedSubscript:(id <NSCopying>)key __attribute__((availability(macosx,introduced=10.8)));

@end

@interface NSOrderedSet : NSObject <NSFastEnumeration>
@end
@interface NSOrderedSet (NSOrderedSetCreation)
- (NSUInteger)count;
@end

@interface NSString : NSObject <NSCopying, NSMutableCopying, NSSecureCoding>

@end

@interface NSNull : NSObject <NSCopying, NSSecureCoding>
+ (NSNull *)null;
@end

// NSMutableArray API
void testNilArgNSMutableArray1() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray addObject:0]; // expected-warning {{Argument to 'NSMutableArray' method 'addObject:' cannot be nil}}
}

void testNilArgNSMutableArray2() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray insertObject:0 atIndex:1]; // expected-warning {{Argument to 'NSMutableArray' method 'insertObject:atIndex:' cannot be nil}}
}

void testNilArgNSMutableArray3() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray replaceObjectAtIndex:1 withObject:0]; // expected-warning {{Argument to 'NSMutableArray' method 'replaceObjectAtIndex:withObject:' cannot be nil}}
}

void testNilArgNSMutableArray4() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  [marray setObject:0 atIndexedSubscript:1]; // expected-warning {{Argument to 'NSMutableArray' method 'setObject:atIndexedSubscript:' cannot be nil}}
}

void testNilArgNSMutableArray5() {
  NSMutableArray *marray = [[NSMutableArray alloc] init];
  marray[1] = 0; // expected-warning {{Array element cannot be nil}}
}

// NSArray API
void testNilArgNSArray1() {
  NSArray *array = [[NSArray alloc] init];
  NSArray *copyArray = [array arrayByAddingObject:0]; // expected-warning {{Argument to 'NSArray' method 'arrayByAddingObject:' cannot be nil}}
}

// NSMutableDictionary and NSDictionary APIs.
void testNilArgNSMutableDictionary1(NSMutableDictionary *d, NSString* key) {
  [d setObject:0 forKey:key]; // expected-warning {{Value argument to 'setObject:forKey:' cannot be nil}}
}

void testNilArgNSMutableDictionary2(NSMutableDictionary *d, NSObject *obj) {
  [d setObject:obj forKey:0]; // expected-warning {{Key argument to 'setObject:forKey:' cannot be nil}}
}

void testNilArgNSMutableDictionary3(NSMutableDictionary *d) {
  [d removeObjectForKey:0]; // expected-warning {{Value argument to 'removeObjectForKey:' cannot be nil}}
}

void testNilArgNSMutableDictionary5(NSMutableDictionary *d, NSString* key) {
  d[key] = 0; // no-warning - removing the mapping for the given key
}
void testNilArgNSMutableDictionary6(NSMutableDictionary *d, NSString *key) {
  if (key)
    ;
  d[key] = 0; // expected-warning {{'NSMutableDictionary' key cannot be nil}}
}

NSDictionary *testNilArgNSDictionary1(NSString* key) {
  return [NSDictionary dictionaryWithObject:0 forKey:key]; // expected-warning {{Value argument to 'dictionaryWithObject:forKey:' cannot be nil}}
}
NSDictionary *testNilArgNSDictionary2(NSObject *obj) {
  return [NSDictionary dictionaryWithObject:obj forKey:0]; // expected-warning {{Key argument to 'dictionaryWithObject:forKey:' cannot be nil}}
}

id testCreateDictionaryLiteralKey(id value, id nilKey) {
  if (nilKey)
    ;
  return @{@"abc":value, nilKey:@"abc"}; // expected-warning {{Dictionary key cannot be nil}}
}

id testCreateDictionaryLiteralValue(id nilValue) {
  if (nilValue)
    ;
  return @{@"abc":nilValue}; // expected-warning {{Dictionary value cannot be nil}}
}

id testCreateDictionaryLiteral(id nilValue, id nilKey) {
  if (nilValue)
    ;
  if (nilKey)
    ;
  return @{@"abc":nilValue, nilKey:@"abc"}; // expected-warning {{Dictionary key cannot be nil}}
                                            // expected-warning@-1 {{Dictionary value cannot be nil}}
}

id testCreateArrayLiteral(id myNil) {
  if (myNil)
    ;
  return @[ @"a", myNil, @"c" ]; // expected-warning {{Array element cannot be nil}}
}

// Test inline defensive checks suppression.
void idc(id x) {
  if (x)
    ;
}
void testIDC(NSMutableDictionary *d, NSString *key) {
  idc(key);
  d[key] = @"abc"; // no-warning
}

@interface Foo {
@public
  int x;
}
- (int *)getPtr;
- (int)getInt;
- (NSMutableDictionary *)getDictPtr;
@property (retain, readonly, nonatomic) Foo* data;
- (NSString*) stringForKeyFE: (id<NSCopying>)key;
@end

void idc2(id x) {
	if (!x)
		return;
}
Foo *retNil() {
  return 0;
}

void testIDC2(Foo *obj) {
	idc2(obj);
	*[obj getPtr] = 1; // no-warning
}

int testIDC3(Foo *obj) {
	idc2(obj);
  return 1/[obj getInt];
}

void testNilReceiverIDC(Foo *obj, NSString *key) {
	NSMutableDictionary *D = [obj getDictPtr];
  idc(D);
  D[key] = @"abc"; // no-warning
}

void testNilReceiverRetNil2(NSMutableDictionary *D, Foo *FooPtrIn, id value) {
  NSString* const kKeyIdentifier = @"key";
	Foo *FooPtr = retNil();
  NSString *key = [[FooPtr data] stringForKeyFE: kKeyIdentifier];
  // key is nil because FooPtr is nil. However, FooPtr is set to nil inside an
  // inlined function, so this error report should be suppressed.
  [D setObject: value forKey: key]; // no-warning
}

void testAssumeNSNullNullReturnsNonNil(NSMutableDictionary *Table, id Object,
                                      id InValue) {
  id Value = Object ? [Table objectForKey:Object] : [NSNull null];
  if (!Value) {
    Value = InValue;
    [Table setObject:Value forKey:Object]; // no warning
  }
}

void testCollectionIsNotEmptyWhenCountIsGreaterThanZero(NSMutableDictionary *D){
  if ([D count] > 0) { // Count is greater than zero.
    NSString *s = 0;
    for (NSString *key in D) {
      s = key;       // Loop is always entered.
    }
    [D removeObjectForKey:s]; // no warning
  }
}

void testCountAwareNSOrderedSet(NSOrderedSet *containers, int *validptr) {
	int *x = 0;
  NSUInteger containerCount = [containers count];
  if (containerCount > 0)    
		x = validptr;
	for (id c in containers) {
		*x = 1; // no warning
	}
}

void testLiteralsNonNil() {
  clang_analyzer_eval(!!@[]); // expected-warning{{TRUE}}
  clang_analyzer_eval(!!@{}); // expected-warning{{TRUE}}
}

@interface NSMutableArray (MySafeAdd)
- (void)addObject:(id)obj safe:(BOOL)safe;
@end

void testArrayCategory(NSMutableArray *arr) {
  [arr addObject:0 safe:1]; // no-warning
}

@interface MyView : NSObject
-(NSArray *)subviews;
@end

void testNoReportWhenReceiverNil(NSMutableArray *array, int b) {
  // Don't warn about adding nil to a container when the receiver is also
  // definitely nil.
  if (array == 0) {
    [array addObject:0]; // no-warning
  }

  MyView *view = b ? [[MyView alloc] init] : 0;
  NSMutableArray *subviews = [[view subviews] mutableCopy];
  // When view is nil, subviews is also nil so there should be no warning
  // here either.
  [subviews addObject:view]; // no-warning
}

NSString *getStringFromString(NSString *string) {
  if (!string)
    return nil;
  return @"New String";
}
void testInlinedDefensiveCheck(NSMutableDictionary *dict, id obj) {
  // The check in getStringFromString() is not a good indication
  // that 'obj' can be nil in this context.
  dict[obj] = getStringFromString(obj); // no-warning
}
