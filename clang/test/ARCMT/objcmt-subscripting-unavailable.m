// RUN: rm -rf %t
// RUN: %clang_cc1 -objcmt-migrate-literals -objcmt-migrate-subscripting -mt-migrate-directory %t %s -x objective-c -triple x86_64-apple-darwin11 
// RUN: c-arcmt-test -mt-migrate-directory %t | arcmt-test -verify-transformed-files %s.result
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s.result

typedef signed char BOOL;
#define nil ((void*) 0)

@interface NSObject
+ (id)alloc;
@end

@interface NSArray : NSObject
- (id)objectAtIndex:(unsigned long)index;
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

@interface NSDictionary : NSObject
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

- (id)objectForKey:(id)aKey;
@end

@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)anObject forKey:(id)aKey;
@end

@interface I
@end
@implementation I
-(void) foo {
  id str;
  NSArray *arr;
  NSDictionary *dict;

  arr = [NSArray array];
  arr = [NSArray arrayWithObject:str];
  arr = [NSArray arrayWithObjects:str, str, nil];
  dict = [NSDictionary dictionary];
  dict = [NSDictionary dictionaryWithObject:arr forKey:str];

  id o = [arr objectAtIndex:2];
  o = [dict objectForKey:@"key"];
  NSMutableArray *marr = 0;
  NSMutableDictionary *mdict = 0;
  [marr replaceObjectAtIndex:2 withObject:@"val"];
  [mdict setObject:@"value" forKey:@"key"];
  [marr replaceObjectAtIndex:2 withObject:[arr objectAtIndex:4]];
  [mdict setObject:[dict objectForKey:@"key2"] forKey:@"key"];
}
@end
