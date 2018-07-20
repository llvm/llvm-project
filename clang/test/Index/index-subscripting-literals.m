@interface NSArray
- (id)objectAtIndexedSubscript:(int)index;
+ (id)arrayWithObjects:(id *)objects count:(unsigned)count;
@end

@interface NSMutableArray : NSArray
- (id)objectAtIndexedSubscript:(int)index;
- (void)setObject:(id)object atIndexedSubscript:(int)index;
@end

@interface NSDictionary
- (id)objectForKeyedSubscript:(id)key;
+ (id)dictionaryWithObjects:(id *)objects forKeys:(id *)keys count:(unsigned)count;
@end

@interface NSMutableDictionary : NSDictionary
- (void)setObject:(id)object forKeyedSubscript:(id)key;
@end

@class NSString;

int idx;
id p;

id testArray() {
  NSMutableArray *array;
  array[idx] = p;
  NSArray *arr = @[ p, p ];
  return array[idx];
}

NSString *key;
id newObject, oldObject;

void testDict() {
  NSMutableDictionary *dictionary;
  oldObject = dictionary[key];
  dictionary[key] = newObject;
  NSDictionary *dict = @{ key: newObject, key: oldObject };
}

// RUN: c-index-test -index-file -target x86_64-apple-macosx10.7 %s | FileCheck %s

// CHECK:      [indexEntityReference]: kind: variable | name: idx | USR: c:@idx | lang: C | cursor: DeclRefExpr=idx:22:5 | loc: 27:9
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: p | USR: c:@p | lang: C | cursor: DeclRefExpr=p:23:4 | loc: 27:16 | <parent>:: kind: function | name: testArray | USR: c:@F@testArray | lang: C | container: [testArray:25:4] | refkind: direct | role: ref
// CHECK-NEXT: [indexEntityReference]: kind: objc-instance-method | name: setObject:atIndexedSubscript:
// CHECK-NEXT: [indexEntityReference]: kind: objc-class | name: NSArray
// CHECK-NEXT: [indexEntityReference]: kind: objc-class-method | name: arrayWithObjects:count: 
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: p | USR: c:@p | lang: C | cursor: DeclRefExpr=p:23:4 | loc: 28:21
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: p | USR: c:@p | lang: C | cursor: DeclRefExpr=p:23:4 | loc: 28:24
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: idx | USR: c:@idx | lang: C | cursor: DeclRefExpr=idx:22:5 | loc: 29:16
// CHECK-NEXT: [indexEntityReference]: kind: objc-instance-method | name: objectAtIndexedSubscript:
// CHECK-NEXT: [indexDeclaration]: kind: variable | name: key

// CHECK:      [indexEntityReference]: kind: variable | name: oldObject
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: key | USR: c:@key | lang: C | cursor: DeclRefExpr=key:32:11 | loc: 37:26
// CHECK-NEXT: [indexEntityReference]: kind: objc-instance-method | name: objectForKeyedSubscript:
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: key | USR: c:@key | lang: C | cursor: DeclRefExpr=key:32:11 | loc: 38:14
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: newObject | USR: c:@newObject | lang: C | cursor: DeclRefExpr=newObject:33:4 | loc: 38:21
// CHECK-NEXT: [indexEntityReference]: kind: objc-instance-method | name: setObject:forKeyedSubscript:
// CHECK-NEXT: [indexEntityReference]: kind: objc-class | name: NSDictionary
// CHECK-NEXT: [indexEntityReference]: kind: objc-class-method | name: dictionaryWithObjects:forKeys:count: 
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: key | USR: c:@key | lang: C | cursor: DeclRefExpr=key:32:11 | loc: 39:27
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: newObject | USR: c:@newObject | lang: C | cursor: DeclRefExpr=newObject:33:4 | loc: 39:32
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: key | USR: c:@key | lang: C | cursor: DeclRefExpr=key:32:11 | loc: 39:43
// CHECK-NEXT: [indexEntityReference]: kind: variable | name: oldObject | USR: c:@oldObject | lang: C | cursor: DeclRefExpr=oldObject:33:15 | loc: 39:48
