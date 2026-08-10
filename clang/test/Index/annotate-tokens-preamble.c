// A comment line.

void f(void *ptr) {
}


// RUN: c-index-test -test-annotate-tokens=%s:1:1:5:1 %s | FileCheck %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-annotate-tokens=%s:1:1:5:1 %s | FileCheck %s
// CHECK: Comment: "// A comment line." [1:1 - 1:19]
// CHECK: Keyword: "void" [3:1 - 3:5] FunctionDecl=f:3:6 (Definition)
// CHECK: Identifier: "f" [3:6 - 3:7] FunctionDecl=f:3:6 (Definition)
// CHECK: Punctuation: "(" [3:7 - 3:8] FunctionDecl=f:3:6 (Definition)
// CHECK: Keyword: "void" [3:8 - 3:12] ParmDecl=ptr:3:14 (Definition)
// CHECK: Punctuation: "*" [3:13 - 3:14] ParmDecl=ptr:3:14 (Definition)
// CHECK: Identifier: "ptr" [3:14 - 3:17] ParmDecl=ptr:3:14 (Definition)
// CHECK: Punctuation: ")" [3:17 - 3:18] FunctionDecl=f:3:6 (Definition)
// CHECK: Punctuation: "{" [3:19 - 3:20] CompoundStmt=
// CHECK: Punctuation: "}" [4:1 - 4:2] CompoundStmt=


