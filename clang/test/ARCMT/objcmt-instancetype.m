// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-instancetype -mt-migrate-directory %t %s -x objective-c -fobjc-runtime-has-weak -fobjc-arc -triple x86_64-apple-darwin11
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fsyntax-only -x objective-c -fobjc-runtime-has-weak -fobjc-arc %s.result

typedef signed char BOOL;
#define nil ((void*) 0)

@interface NSObject
+ (id)alloc;
@end

@interface NSString : NSObject
+ (id)stringWithString:(NSString *)string;
- (id)initWithString:(NSString *)aString;
@end

@implementation NSString : NSObject
+ (id)stringWithString:(NSString *)string { return 0; };
- (instancetype)initWithString:(NSString *)aString { return 0; };
@end

@interface NSArray : NSObject
- (id)objectAtIndex:(unsigned long)index;
- (id)objectAtIndexedSubscript:(int)index;
@end

@interface NSArray (NSArrayCreation)
+ (id)array;
+ (id)arrayWithObject:(id)anObject;
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
+ (id)arrayWithObjects:(id)firstObj, ...;
+ arrayWithArray:(NSArray *)array;

- (id)initWithObjects:(const id [])objects count:(unsigned long)cnt;
- (id)initWithObjects:(id)firstObj, ...;
- (id)initWithArray:(NSArray *)array;

- (id)objectAtIndex:(unsigned long)index;
@end

@implementation NSArray (NSArrayCreation)
+ (id)array { return 0; }
+ (id)arrayWithObject:(id)anObject {
  return anObject;
}
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt { return 0; }
+ (id)arrayWithObjects:(id)firstObj, ... {
  return 0; }
+ arrayWithArray:(NSArray *)array {
  return 0;
}

- (id)initWithObjects:(const id [])objects count:(unsigned long)cnt { return 0; }
- (id)initWithObjects:(id)firstObj, ... { return 0; }
- (id)initWithArray:(NSArray *)array { return 0; }

- (id)objectAtIndex:(unsigned long)index { return 0; }
@end

@interface NSMutableArray : NSArray
- (void)replaceObjectAtIndex:(unsigned long)index withObject:(id)anObject;
- (void)setObject:(id)object atIndexedSubscript:(int)index;
@end

@interface NSDictionary : NSObject
- (id)objectForKeyedSubscript:(id)key;
@end

@interface NSDictionary (NSDictionaryCreation)
+ (id)dictionary;
+ (id)dictionaryWithObject:(id)object forKey:(id)key;
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
+ dictionaryWithObjectsAndKeys:(id)firstObject, ...;
+ (id)dictionaryWithDictionary:(NSDictionary *)dict;
+ (id)dictionaryWithObjects:(NSArray *)objects forKeys:(NSArray *)keys;

- (id)initWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
- (id)initWithObjectsAndKeys:(id)firstObject, ...;
- (id)initWithDictionary:(NSDictionary *)otherDictionary;
- (id)initWithObjects:(NSArray *)objects forKeys:(NSArray *)keys;

- (id)objectForKey:(id)aKey;
@end

@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)anObject forKey:(id)aKey;
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

@interface NSNumber : NSObject
@end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithInt:(int)value;
@end

@implementation NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithInt:(int)value { return 0; }
@end

#define M(x) (x)
#define PAIR(x) @#x, [NSNumber numberWithInt:(x)]
#define TWO(x) ((x), (x))

void foo(void) {
  NSString *str = M([NSString stringWithString:@"foo"]); // expected-warning {{redundant}}
  str = [[NSString alloc] initWithString:@"foo"]; // expected-warning {{redundant}}
  NSArray *arr = [NSArray arrayWithArray:@[str]]; // expected-warning {{redundant}}
  NSDictionary *dict = [NSDictionary dictionaryWithDictionary:@{str: arr}]; // expected-warning {{redundant}}
}
