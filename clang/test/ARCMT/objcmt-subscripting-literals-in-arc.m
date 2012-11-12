// RUN: rm -rf %t
// RUN: %clang_cc1 -fobjc-arc -objcmt-migrate-literals -objcmt-migrate-subscripting -mt-migrate-directory %t %s -x objective-c -triple x86_64-apple-darwin11 
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -fobjc-arc -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s.result

typedef signed char BOOL;
#define nil ((void*) 0)

typedef const struct __CFString * CFStringRef;

@interface NSObject
+ (id)alloc;
@end

@protocol NSCopying
@end

@interface NSString : NSObject
+ (id)stringWithString:(NSString *)string;
- (id)initWithString:(NSString *)aString;
@end

@interface NSArray : NSObject
- (id)objectAtIndex:(unsigned long)index;
@end

@interface NSArray (NSExtendedArray)
- (id)objectAtIndexedSubscript:(unsigned)idx;
@end

@interface NSArray (NSArrayCreation)
+ (id)array;
+ (id)arrayWithObject:(id)anObject;
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
+ (id)arrayWithObjects:(id)firstObj, ...;
+ (id)arrayWithArray:(NSArray *)array;

- (id)initWithObjects:(const id [])objects count:(unsigned long)cnt;
- (id)initWithObjects:(id)firstObj, ...;
- (id)initWithArray:(NSArray *)array;
@end

@interface NSMutableArray : NSArray
- (void)replaceObjectAtIndex:(unsigned long)index withObject:(id)anObject;
@end

@interface NSMutableArray (NSExtendedMutableArray)
- (void)setObject:(id)obj atIndexedSubscript:(unsigned)idx;
@end

@interface NSDictionary : NSObject
- (id)objectForKey:(id)aKey;
@end

@interface NSDictionary (NSExtendedDictionary)
- (id)objectForKeyedSubscript:(id)key;
@end

@interface NSDictionary (NSDictionaryCreation)
+ (id)dictionary;
+ (id)dictionaryWithObject:(id)object forKey:(id)key;
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
+ (id)dictionaryWithObjectsAndKeys:(id)firstObject, ...;
+ (id)dictionaryWithDictionary:(NSDictionary *)dict;
+ (id)dictionaryWithObjects:(NSArray *)objects forKeys:(NSArray *)keys;

- (id)initWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
- (id)initWithObjectsAndKeys:(id)firstObject, ...;
- (id)initWithDictionary:(NSDictionary *)otherDictionary;
- (id)initWithObjects:(NSArray *)objects forKeys:(NSArray *)keys;
@end

@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)anObject forKey:(id)aKey;
@end

@interface NSMutableDictionary (NSExtendedMutableDictionary)
- (void)setObject:(id)obj forKeyedSubscript:(id <NSCopying>)key;
@end

@interface NSNumber : NSObject
@end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithInt:(int)value;
- (id)initWithInt:(int)value;
@end

@interface I {
  NSArray *ivarArr;
}
@end
@implementation I
-(void) foo {
  NSString *str;
  NSArray *arr;
  NSDictionary *dict;

  arr = [NSArray arrayWithObjects:str, str, nil];
  arr = [[NSArray alloc] initWithObjects:str, str, nil];
  dict = [NSDictionary dictionaryWithObjectsAndKeys: @"value1", @"key1", @"value2", @"key2", nil];
  dict = [[NSDictionary alloc] initWithObjectsAndKeys: @"value1", @"key1", @"value2", @"key2", nil];

  NSNumber *n = [[NSNumber alloc] initWithInt:2];
}
@end
