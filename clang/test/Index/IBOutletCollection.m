#define IBOutletCollection(ClassName) __attribute__((iboutletcollection(ClassName)))

@interface Test {
  IBOutletCollection(Test) Test *anOutletCollection;
}
@end

// RUN: c-index-test -cursor-at=%s:4:24 %s | FileCheck -check-prefix=CHECK-CURSOR %s
// CHECK-CURSOR: ObjCClassRef=Test:3:12

// RUN: c-index-test -test-annotate-tokens=%s:4:1:5:1 %s | FileCheck -check-prefix=CHECK-TOK %s
// CHECK-TOK: Identifier: "IBOutletCollection" [4:3 - 4:21] macro expansion=IBOutletCollection:1:9
// FIXME: The following token should belong to the macro expansion cursor.
// CHECK-TOK: Punctuation: "(" [4:21 - 4:22] attribute(iboutletcollection)= [IBOutletCollection=ObjCInterface]
// CHECK-TOK: Identifier: "Test" [4:22 - 4:26] ObjCClassRef=Test:3:12
// FIXME: The following token should belong to the macro expansion cursor.
// CHECK-TOK: Punctuation: ")" [4:26 - 4:27]
// CHECK-TOK: Identifier: "Test" [4:28 - 4:32] ObjCClassRef=Test:3:12
// CHECK-TOK: Punctuation: "*" [4:33 - 4:34] ObjCIvarDecl=anOutletCollection:4:34 (Definition)
// CHECK-TOK: Identifier: "anOutletCollection" [4:34 - 4:52] ObjCIvarDecl=anOutletCollection:4:34 (Definition)
