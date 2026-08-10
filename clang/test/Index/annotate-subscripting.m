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

// RUN: c-index-test -test-annotate-tokens=%s:22:1:36:1 -target x86_64-apple-macosx10.7 %s | FileCheck %s
// CHECK: Identifier: "array" [24:3 - 24:8] DeclRefExpr=array:23:19
// CHECK: Punctuation: "[" [24:8 - 24:9] UnexposedExpr=
// CHECK: Literal: "3" [24:9 - 24:10] IntegerLiteral=
// CHECK: Punctuation: "]" [24:10 - 24:11] UnexposedExpr=
// CHECK: Punctuation: "=" [24:12 - 24:13] BinaryOperator=
// CHECK: Literal: "0" [24:14 - 24:15] IntegerLiteral=
// CHECK: Punctuation: ";" [24:15 - 24:16] CompoundStmt=
// CHECK: Identifier: "NSArray" [25:3 - 25:10] ObjCClassRef=NSArray:1:12
// CHECK: Punctuation: "*" [25:11 - 25:12] VarDecl=arr:25:12 (Definition)
// CHECK: Identifier: "arr" [25:12 - 25:15] VarDecl=arr:25:12 (Definition)
// CHECK: Punctuation: "=" [25:16 - 25:17] VarDecl=arr:25:12 (Definition)
// CHECK: Punctuation: "@" [25:18 - 25:19] UnexposedExpr=
// CHECK: Punctuation: "[" [25:19 - 25:20] UnexposedExpr=
// CHECK: Identifier: "p" [25:21 - 25:22] DeclRefExpr=p:22:28
// CHECK: Punctuation: "," [25:22 - 25:23] UnexposedExpr=
// CHECK: Identifier: "p" [25:24 - 25:25] DeclRefExpr=p:22:28
// CHECK: Punctuation: "]" [25:26 - 25:27] UnexposedExpr=
// CHECK: Punctuation: ";" [25:27 - 25:28] DeclStmt=
// CHECK: Keyword: "return" [26:3 - 26:9] ReturnStmt=
// CHECK: Identifier: "array" [26:10 - 26:15] DeclRefExpr=array:23:19
// CHECK: Punctuation: "[" [26:15 - 26:16] UnexposedExpr=
// CHECK: Identifier: "index" [26:16 - 26:21] DeclRefExpr=index:22:18
// CHECK: Punctuation: "]" [26:21 - 26:22] UnexposedExpr=
// CHECK: Punctuation: ";" [26:22 - 26:23] CompoundStmt=

// CHECK: Identifier: "oldObject" [33:3 - 33:12] DeclRefExpr=oldObject:32:17
// CHECK: Punctuation: "=" [33:13 - 33:14] BinaryOperator=
// CHECK: Identifier: "dictionary" [33:15 - 33:25] DeclRefExpr=dictionary:30:24
// CHECK: Punctuation: "[" [33:25 - 33:26] UnexposedExpr=
// CHECK: Identifier: "key" [33:26 - 33:29] DeclRefExpr=key:31:13
// CHECK: Punctuation: "]" [33:29 - 33:30] UnexposedExpr=
// CHECK: Punctuation: ";" [33:30 - 33:31] CompoundStmt=
// CHECK: Identifier: "dictionary" [34:3 - 34:13] DeclRefExpr=dictionary:30:24
// CHECK: Punctuation: "[" [34:13 - 34:14] UnexposedExpr=
// CHECK: Identifier: "key" [34:14 - 34:17] DeclRefExpr=key:31:13
// CHECK: Punctuation: "]" [34:17 - 34:18] UnexposedExpr=
// CHECK: Punctuation: "=" [34:19 - 34:20] BinaryOperator=
// CHECK: Identifier: "newObject" [34:21 - 34:30] DeclRefExpr=newObject:32:6
// CHECK: Punctuation: ";" [34:30 - 34:31] CompoundStmt=
// CHECK: Identifier: "NSDictionary" [35:3 - 35:15] ObjCClassRef=NSDictionary:11:12
// CHECK: Punctuation: "*" [35:16 - 35:17] VarDecl=dict:35:17 (Definition)
// CHECK: Identifier: "dict" [35:17 - 35:21] VarDecl=dict:35:17 (Definition)
// CHECK: Punctuation: "=" [35:22 - 35:23] VarDecl=dict:35:17 (Definition)
// CHECK: Punctuation: "@" [35:24 - 35:25] UnexposedExpr=
// CHECK: Punctuation: "{" [35:25 - 35:26] UnexposedExpr=
// CHECK: Identifier: "key" [35:27 - 35:30] DeclRefExpr=key:31:13
// CHECK: Punctuation: ":" [35:30 - 35:31] UnexposedExpr=
// CHECK: Identifier: "newObject" [35:32 - 35:41] DeclRefExpr=newObject:32:6
// CHECK: Punctuation: "," [35:41 - 35:42] UnexposedExpr=
// CHECK: Identifier: "key" [35:43 - 35:46] DeclRefExpr=key:31:13
// CHECK: Punctuation: ":" [35:46 - 35:47] UnexposedExpr=
// CHECK: Identifier: "oldObject" [35:48 - 35:57] DeclRefExpr=oldObject:32:17
// CHECK: Punctuation: "}" [35:58 - 35:59] UnexposedExpr=
// CHECK: Punctuation: ";" [35:59 - 35:60] DeclStmt=
