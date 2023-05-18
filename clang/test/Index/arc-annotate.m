@interface A
@property (strong, nonatomic) id property;
@property (nonatomic, weak) id second_property;
@property (unsafe_unretained, nonatomic) id third_property;
@end

void foo() {
  A *avar;
  avar = 0;
}

// RUN: c-index-test -test-annotate-tokens=%s:1:1:11:1 %s -fobjc-arc -fobjc-nonfragile-abi | FileCheck %s
// CHECK: Punctuation: "@" [1:1 - 1:2] ObjCInterfaceDecl=A:1:12
// CHECK: Keyword: "interface" [1:2 - 1:11] ObjCInterfaceDecl=A:1:12
// CHECK: Identifier: "A" [1:12 - 1:13] ObjCInterfaceDecl=A:1:12
// CHECK: Punctuation: "@" [2:1 - 2:2] ObjCPropertyDecl=property:2:34
// CHECK: Keyword: "property" [2:2 - 2:10] ObjCPropertyDecl=property:2:34
// CHECK: Punctuation: "(" [2:11 - 2:12] ObjCPropertyDecl=property:2:34
// CHECK: Keyword: "strong" [2:12 - 2:18] ObjCPropertyDecl=property:2:34
// CHECK: Punctuation: "," [2:18 - 2:19] ObjCPropertyDecl=property:2:34
// CHECK: Keyword: "nonatomic" [2:20 - 2:29] ObjCPropertyDecl=property:2:34
// CHECK: Punctuation: ")" [2:29 - 2:30] ObjCPropertyDecl=property:2:34
// CHECK: Identifier: "id" [2:31 - 2:33] TypeRef=id:0:0
// CHECK: Identifier: "property" [2:34 - 2:42] ObjCPropertyDecl=property:2:34
// CHECK: Punctuation: ";" [2:42 - 2:43] ObjCInterfaceDecl=A:1:12
// CHECK: Punctuation: "@" [3:1 - 3:2] ObjCPropertyDecl=second_property:3:32
// CHECK: Keyword: "property" [3:2 - 3:10] ObjCPropertyDecl=second_property:3:32
// CHECK: Punctuation: "(" [3:11 - 3:12] ObjCPropertyDecl=second_property:3:32
// CHECK: Keyword: "nonatomic" [3:12 - 3:21] ObjCPropertyDecl=second_property:3:32
// CHECK: Punctuation: "," [3:21 - 3:22] ObjCPropertyDecl=second_property:3:32
// CHECK: Keyword: "weak" [3:23 - 3:27] ObjCPropertyDecl=second_property:3:32
// CHECK: Punctuation: ")" [3:27 - 3:28] ObjCPropertyDecl=second_property:3:32
// CHECK: Identifier: "id" [3:29 - 3:31] TypeRef=id:0:0
// CHECK: Identifier: "second_property" [3:32 - 3:47] ObjCPropertyDecl=second_property:3:32
// CHECK: Punctuation: "@" [4:1 - 4:2] ObjCPropertyDecl=third_property:4:45
// CHECK: Keyword: "property" [4:2 - 4:10] ObjCPropertyDecl=third_property:4:45
// CHECK: Punctuation: "(" [4:11 - 4:12] ObjCPropertyDecl=third_property:4:45
// CHECK: Keyword: "unsafe_unretained" [4:12 - 4:29] ObjCPropertyDecl=third_property:4:45
// CHECK: Punctuation: "," [4:29 - 4:30] ObjCPropertyDecl=third_property:4:45
// CHECK: Keyword: "nonatomic" [4:31 - 4:40] ObjCPropertyDecl=third_property:4:45
// CHECK: Punctuation: ")" [4:40 - 4:41] ObjCPropertyDecl=third_property:4:45
// CHECK: Identifier: "id" [4:42 - 4:44] TypeRef=id:0:0
// CHECK: Identifier: "third_property" [4:45 - 4:59] ObjCPropertyDecl=third_property:4:45

// CHECK: Identifier: "A" [8:3 - 8:4] ObjCClassRef=A:1:12
// CHECK: Punctuation: "*" [8:5 - 8:6] VarDecl=avar:8:6 (Definition)
// CHECK: Identifier: "avar" [8:6 - 8:10] VarDecl=avar:8:6 (Definition)
// CHECK: Punctuation: ";" [8:10 - 8:11] DeclStmt=
// CHECK: Identifier: "avar" [9:3 - 9:7] DeclRefExpr=avar:8:6
// CHECK: Punctuation: "=" [9:8 - 9:9] BinaryOperator=
// CHECK: Literal: "0" [9:10 - 9:11] IntegerLiteral=
// CHECK: Punctuation: ";" [9:11 - 9:12] CompoundStmt=

// RUN: c-index-test -file-refs-at=%s:8:8 %s -fobjc-arc -fobjc-nonfragile-abi | FileCheck %s -check-prefix=CHECK-REFS
// CHECK-REFS: VarDecl=avar:8:6 (Definition)
// CHECK-REFS: VarDecl=avar:8:6 (Definition) =[8:6 - 8:10]
// CHECK-REFS: DeclRefExpr=avar:8:6 =[9:3 - 9:7]
