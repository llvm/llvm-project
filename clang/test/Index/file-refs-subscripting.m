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

id testArray(int index, id p) {
  NSMutableArray *array;
  array[3] = 0;
  NSArray *arr = @[ p, p ];
  return array[index];
}

void testDict() {
  NSMutableDictionary *dictionary;
  NSString *key;
  id newObject, oldObject;
  oldObject = dictionary[key];
  dictionary[key] = newObject;
  NSDictionary *dict = @{ key: newObject, key: oldObject };
}

// RUN: c-index-test \

// RUN:  -file-refs-at=%s:22:21 \
// CHECK:      ParmDecl=index:22:18
// CHECK-NEXT: ParmDecl=index:22:18 (Definition) =[22:18 - 22:23]
// CHECK-NEXT: DeclRefExpr=index:22:18 =[26:16 - 26:21]

// RUN:  -file-refs-at=%s:22:28 \
// CHECK-NEXT: ParmDecl=p:22:28
// CHECK-NEXT: ParmDecl=p:22:28 (Definition) =[22:28 - 22:29]
// CHECK-NEXT: DeclRefExpr=p:22:28 =[25:21 - 25:22]
// CHECK-NEXT: DeclRefExpr=p:22:28 =[25:24 - 25:25]

// RUN:  -file-refs-at=%s:34:16 \
// CHECK-NEXT: DeclRefExpr=key:31:13
// CHECK-NEXT: VarDecl=key:31:13 (Definition) =[31:13 - 31:16]
// CHECK-NEXT: DeclRefExpr=key:31:13 =[33:26 - 33:29]
// CHECK-NEXT: DeclRefExpr=key:31:13 =[34:14 - 34:17]
// CHECK-NEXT: DeclRefExpr=key:31:13 =[35:27 - 35:30]
// CHECK-NEXT: DeclRefExpr=key:31:13 =[35:43 - 35:46]

// RUN:  -file-refs-at=%s:35:35 \
// CHECK-NEXT: DeclRefExpr=newObject:32:6
// CHECK-NEXT: VarDecl=newObject:32:6 (Definition) =[32:6 - 32:15]
// CHECK-NEXT: DeclRefExpr=newObject:32:6 =[34:21 - 34:30]
// CHECK-NEXT: DeclRefExpr=newObject:32:6 =[35:32 - 35:41]

// RUN:   -target x86_64-apple-macosx10.7 %s | FileCheck %s
