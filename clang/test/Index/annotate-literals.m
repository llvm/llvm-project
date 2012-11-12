typedef unsigned char BOOL;

@interface NSNumber @end

@interface NSNumber (NSNumberCreation)
+ (NSNumber *)numberWithChar:(char)value;
+ (NSNumber *)numberWithUnsignedChar:(unsigned char)value;
+ (NSNumber *)numberWithShort:(short)value;
+ (NSNumber *)numberWithUnsignedShort:(unsigned short)value;
+ (NSNumber *)numberWithInt:(int)value;
+ (NSNumber *)numberWithUnsignedInt:(unsigned int)value;
+ (NSNumber *)numberWithLong:(long)value;
+ (NSNumber *)numberWithUnsignedLong:(unsigned long)value;
+ (NSNumber *)numberWithLongLong:(long long)value;
+ (NSNumber *)numberWithUnsignedLongLong:(unsigned long long)value;
+ (NSNumber *)numberWithFloat:(float)value;
+ (NSNumber *)numberWithDouble:(double)value;
+ (NSNumber *)numberWithBool:(BOOL)value;
@end

@interface NSArray
@end

@interface NSArray (NSArrayCreation)
+ (id)arrayWithObjects:(const id [])objects count:(unsigned long)cnt;
@end

@interface NSDictionary
+ (id)dictionaryWithObjects:(const id [])objects forKeys:(const id [])keys count:(unsigned long)cnt;
@end

void test_literals(id k1, id o1, id k2, id o2, id k3) {
  id objects = @[ o1, o2 ];
  id dict = @{ k1 : o1,
               k2 : o2,
               k3 : @17 };
}


// RUN: c-index-test -test-annotate-tokens=%s:33:1:37:1 %s | FileCheck -check-prefix=CHECK-LITERALS %s

// CHECK-LITERALS: Identifier: "id" [33:3 - 33:5] TypeRef=id:0:0
// CHECK-LITERALS: Identifier: "objects" [33:6 - 33:13] VarDecl=objects:33:6 (Definition)
// CHECK-LITERALS: Punctuation: "=" [33:14 - 33:15] VarDecl=objects:33:6 (Definition)
// CHECK-LITERALS: Punctuation: "@" [33:16 - 33:17] UnexposedExpr=
// CHECK-LITERALS: Punctuation: "[" [33:17 - 33:18] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o1" [33:19 - 33:21] DeclRefExpr=o1:32:30
// CHECK-LITERALS: Punctuation: "," [33:21 - 33:22] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o2" [33:23 - 33:25] DeclRefExpr=o2:32:44
// CHECK-LITERALS: Punctuation: "]" [33:26 - 33:27] UnexposedExpr=
// CHECK-LITERALS: Punctuation: ";" [33:27 - 33:28] DeclStmt=
// CHECK-LITERALS: Identifier: "id" [34:3 - 34:5] TypeRef=id:0:0
// CHECK-LITERALS: Identifier: "dict" [34:6 - 34:10] VarDecl=dict:34:6 (Definition)
// CHECK-LITERALS: Punctuation: "=" [34:11 - 34:12] VarDecl=dict:34:6 (Definition)
// CHECK-LITERALS: Punctuation: "@" [34:13 - 34:14] UnexposedExpr=
// CHECK-LITERALS: Punctuation: "{" [34:14 - 34:15] UnexposedExpr=
// CHECK-LITERALS: Identifier: "k1" [34:16 - 34:18] DeclRefExpr=k1:32:23
// CHECK-LITERALS: Punctuation: ":" [34:19 - 34:20] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o1" [34:21 - 34:23] DeclRefExpr=o1:32:30
// CHECK-LITERALS: Punctuation: "," [34:23 - 34:24] UnexposedExpr=
// CHECK-LITERALS: Identifier: "k2" [35:16 - 35:18] DeclRefExpr=k2:32:37
// CHECK-LITERALS: Punctuation: ":" [35:19 - 35:20] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o2" [35:21 - 35:23] DeclRefExpr=o2:32:44
// CHECK-LITERALS: Punctuation: "," [35:23 - 35:24] UnexposedExpr=
// CHECK-LITERALS: Identifier: "k3" [36:16 - 36:18] DeclRefExpr=k3:32:51
// CHECK-LITERALS: Punctuation: ":" [36:19 - 36:20] UnexposedExpr=
// CHECK-LITERALS: Punctuation: "@" [36:21 - 36:22] UnexposedExpr=
// CHECK-LITERALS: Literal: "17" [36:22 - 36:24] IntegerLiteral=
// CHECK-LITERALS: Punctuation: "}" [36:25 - 36:26] UnexposedExpr=
// CHECK-LITERALS: Punctuation: ";" [36:26 - 36:27] DeclStmt=
// CHECK-LITERALS: Punctuation: "}" [37:1 - 37:2] CompoundStmt=

