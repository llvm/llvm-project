// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o %t.nopch.ll %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-pch -o %t.pch %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o %t.pch.ll %s -include-pch %t.pch
// RUN: diff %t.nopch.ll %t.pch.ll

#ifndef HEADER
#define HEADER

@interface NSArray
- (id)objectAtIndexedSubscript:(int)index;
+ (id)arrayWithObjects:(id *)objects count:(unsigned)count;
@end

@interface NSMutableArray : NSArray
- (void)setObject:(id)object atIndexedSubscript:(int)index;
@end

@interface NSDictionary
- (id)objectForKeyedSubscript:(id)key;
+ (id)dictionaryWithObjects:(id *)objects forKeys:(id *)keys count:(unsigned)count;
@end

@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

@interface NSNumber
+ (NSNumber *)numberWithInt:(int)value;
@end

@class NSString;

id testArray(int idx, id p) {
  NSMutableArray *array;
  array[idx] = p;
  NSArray *arr = @[ p, @7 ];
  return array[idx];
}

void testDict(NSString *key, id newObject, id oldObject) {
  NSMutableDictionary *dictionary;
  oldObject = dictionary[key];
  dictionary[key] = newObject;
  NSDictionary *dict = @{ key: newObject, key: oldObject };
}

#endif
