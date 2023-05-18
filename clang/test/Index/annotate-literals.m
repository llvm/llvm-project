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

@interface NSValue
+ (NSValue *)valueWithBytes:(const void *)value objCType:(const char *)type;
@end

typedef struct __attribute__((objc_boxable)) _c_struct {
  int dummy;
} c_struct;

void test_literals(id k1, id o1, id k2, id o2, id k3, c_struct s) {
  id objects = @[ o1, o2 ];
  id dict = @{ k1 : o1,
               k2 : o2,
               k3 : @17 };
  id val = @(s);
}


// RUN: c-index-test -test-annotate-tokens=%s:41:1:46:1 %s | FileCheck -check-prefix=CHECK-LITERALS %s

// CHECK-LITERALS: Identifier: "id" [41:3 - 41:5] TypeRef=id:0:0
// CHECK-LITERALS: Identifier: "objects" [41:6 - 41:13] VarDecl=objects:41:6 (Definition)
// CHECK-LITERALS: Punctuation: "=" [41:14 - 41:15] VarDecl=objects:41:6 (Definition)
// CHECK-LITERALS: Punctuation: "@" [41:16 - 41:17] UnexposedExpr=
// CHECK-LITERALS: Punctuation: "[" [41:17 - 41:18] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o1" [41:19 - 41:21] DeclRefExpr=o1:40:30
// CHECK-LITERALS: Punctuation: "," [41:21 - 41:22] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o2" [41:23 - 41:25] DeclRefExpr=o2:40:44
// CHECK-LITERALS: Punctuation: "]" [41:26 - 41:27] UnexposedExpr=
// CHECK-LITERALS: Punctuation: ";" [41:27 - 41:28] DeclStmt=
// CHECK-LITERALS: Identifier: "id" [42:3 - 42:5] TypeRef=id:0:0
// CHECK-LITERALS: Identifier: "dict" [42:6 - 42:10] VarDecl=dict:42:6 (Definition)
// CHECK-LITERALS: Punctuation: "=" [42:11 - 42:12] VarDecl=dict:42:6 (Definition)
// CHECK-LITERALS: Punctuation: "@" [42:13 - 42:14] UnexposedExpr=
// CHECK-LITERALS: Punctuation: "{" [42:14 - 42:15] UnexposedExpr=
// CHECK-LITERALS: Identifier: "k1" [42:16 - 42:18] DeclRefExpr=k1:40:23
// CHECK-LITERALS: Punctuation: ":" [42:19 - 42:20] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o1" [42:21 - 42:23] DeclRefExpr=o1:40:30
// CHECK-LITERALS: Punctuation: "," [42:23 - 42:24] UnexposedExpr=
// CHECK-LITERALS: Identifier: "k2" [43:16 - 43:18] DeclRefExpr=k2:40:37
// CHECK-LITERALS: Punctuation: ":" [43:19 - 43:20] UnexposedExpr=
// CHECK-LITERALS: Identifier: "o2" [43:21 - 43:23] DeclRefExpr=o2:40:44
// CHECK-LITERALS: Punctuation: "," [43:23 - 43:24] UnexposedExpr=
// CHECK-LITERALS: Identifier: "k3" [44:16 - 44:18] DeclRefExpr=k3:40:51
// CHECK-LITERALS: Punctuation: ":" [44:19 - 44:20] UnexposedExpr=
// CHECK-LITERALS: Punctuation: "@" [44:21 - 44:22] UnexposedExpr=
// CHECK-LITERALS: Literal: "17" [44:22 - 44:24] IntegerLiteral=
// CHECK-LITERALS: Punctuation: "}" [44:25 - 44:26] UnexposedExpr=
// CHECK-LITERALS: Punctuation: ";" [44:26 - 44:27] DeclStmt=
// CHECK-LITERALS: Identifier: "id" [45:3 - 45:5] TypeRef=id:0:0
// CHECK-LITERALS: Identifier: "val" [45:6 - 45:9] VarDecl=val:45:6 (Definition)
// CHECK-LITERALS: Punctuation: "=" [45:10 - 45:11] VarDecl=val:45:6 (Definition)
// CHECK-LITERALS: Punctuation: "@" [45:12 - 45:13] UnexposedExpr=
// CHECK-LITERALS: Punctuation: "(" [45:13 - 45:14] ParenExpr=
// CHECK-LITERALS: Identifier: "s" [45:14 - 45:15] DeclRefExpr=s:40:64
// CHECK-LITERALS: Punctuation: ")" [45:15 - 45:16] ParenExpr=
// CHECK-LITERALS: Punctuation: ";" [45:16 - 45:17] DeclStmt=
// CHECK-LITERALS: Punctuation: "}" [46:1 - 46:2] CompoundStmt=

