struct Foo {
  int operator[](int key);
  int operator()(int key = 2);
};

void testFoo(Foo foo, int index) {
  foo();
  foo(index);

  foo[index];
  foo[index + index];

  foo[foo[index]];
  foo[foo() + foo[index]];
  foo[foo(index) + foo[index]];
}

// RUN: c-index-test -test-annotate-tokens=%s:7:1:7:100 %s -std=c++11 -Wno-unused-value | FileCheck %s -check-prefix=CHECK1
// CHECK1: Identifier: "foo" [7:3 - 7:6] DeclRefExpr=foo:6:18
// CHECK1: Punctuation: "(" [7:6 - 7:7] CallExpr=operator():3:7
// CHECK1: Punctuation: ")" [7:7 - 7:8] CallExpr=operator():3:7
// CHECK1: Punctuation: ";" [7:8 - 7:9] CompoundStmt=

// RUN: c-index-test -test-annotate-tokens=%s:8:1:8:100 %s -std=c++11 -Wno-unused-value | FileCheck %s -check-prefix=CHECK2
// CHECK2: Punctuation: "(" [8:6 - 8:7] CallExpr=operator():3:7
// CHECK2: Identifier: "index" [8:7 - 8:12] DeclRefExpr=index:6:27
// CHECK2: Punctuation: ")" [8:12 - 8:13] CallExpr=operator():3:7
// CHECK2: Punctuation: ";" [8:13 - 8:14] CompoundStmt=

// RUN: c-index-test -test-annotate-tokens=%s:10:1:10:100 %s -std=c++11 -Wno-unused-value | FileCheck %s -check-prefix=CHECK3
// CHECK3: Identifier: "foo" [10:3 - 10:6] DeclRefExpr=foo:6:18
// CHECK3: Punctuation: "[" [10:6 - 10:7] CallExpr=operator[]:2:7
// CHECK3: Identifier: "index" [10:7 - 10:12] DeclRefExpr=index:6:27
// CHECK3: Punctuation: "]" [10:12 - 10:13] CallExpr=operator[]:2:7
// CHECK3: Punctuation: ";" [10:13 - 10:14] CompoundStmt=

// RUN: c-index-test -test-annotate-tokens=%s:11:1:11:100 %s -std=c++11 -Wno-unused-value | FileCheck %s -check-prefix=CHECK4
// CHECK4: Identifier: "foo" [11:3 - 11:6] DeclRefExpr=foo:6:18
// CHECK4: Punctuation: "[" [11:6 - 11:7] CallExpr=operator[]:2:7
// CHECK4: Identifier: "index" [11:7 - 11:12] DeclRefExpr=index:6:27
// CHECK4: Punctuation: "+" [11:13 - 11:14] BinaryOperator=
// CHECK4: Identifier: "index" [11:15 - 11:20] DeclRefExpr=index:6:27
// CHECK4: Punctuation: "]" [11:20 - 11:21] CallExpr=operator[]:2:7
// CHECK4: Punctuation: ";" [11:21 - 11:22] CompoundStmt=

// RUN: c-index-test -test-annotate-tokens=%s:13:1:13:100 %s -std=c++11 -Wno-unused-value | FileCheck %s -check-prefix=CHECK5
// CHECK5: Identifier: "foo" [13:3 - 13:6] DeclRefExpr=foo:6:18
// CHECK5: Punctuation: "[" [13:6 - 13:7] CallExpr=operator[]:2:7
// CHECK5: Identifier: "foo" [13:7 - 13:10] DeclRefExpr=foo:6:18
// CHECK5: Punctuation: "[" [13:10 - 13:11] CallExpr=operator[]:2:7
// CHECK5: Identifier: "index" [13:11 - 13:16] DeclRefExpr=index:6:27
// CHECK5: Punctuation: "]" [13:16 - 13:17] CallExpr=operator[]:2:7
// CHECK5: Punctuation: "]" [13:17 - 13:18] CallExpr=operator[]:2:7
// CHECK5: Punctuation: ";" [13:18 - 13:19] CompoundStmt=

// RUN: c-index-test -test-annotate-tokens=%s:14:1:14:100 %s -std=c++11 -Wno-unused-value | FileCheck %s -check-prefix=CHECK6
// CHECK6: Identifier: "foo" [14:3 - 14:6] DeclRefExpr=foo:6:18
// CHECK6: Punctuation: "[" [14:6 - 14:7] CallExpr=operator[]:2:7
// CHECK6: Identifier: "foo" [14:7 - 14:10] DeclRefExpr=foo:6:18
// CHECK6: Punctuation: "(" [14:10 - 14:11] CallExpr=operator():3:7
// CHECK6: Punctuation: ")" [14:11 - 14:12] CallExpr=operator():3:7
// CHECK6: Punctuation: "+" [14:13 - 14:14] BinaryOperator=
// CHECK6: Identifier: "foo" [14:15 - 14:18] DeclRefExpr=foo:6:18
// CHECK6: Punctuation: "[" [14:18 - 14:19] CallExpr=operator[]:2:7
// CHECK6: Identifier: "index" [14:19 - 14:24] DeclRefExpr=index:6:27
// CHECK6: Punctuation: "]" [14:24 - 14:25] CallExpr=operator[]:2:7
// CHECK6: Punctuation: "]" [14:25 - 14:26] CallExpr=operator[]:2:7
// CHECK6: Punctuation: ";" [14:26 - 14:27] CompoundStmt=

// RUN: c-index-test -test-annotate-tokens=%s:15:1:15:100 %s -std=c++11 -Wno-unused-value | FileCheck %s -check-prefix=CHECK7
// CHECK7: Identifier: "foo" [15:3 - 15:6] DeclRefExpr=foo:6:18
// CHECK7: Punctuation: "[" [15:6 - 15:7] CallExpr=operator[]:2:7
// CHECK7: Identifier: "foo" [15:7 - 15:10] DeclRefExpr=foo:6:18
// CHECK7: Punctuation: "(" [15:10 - 15:11] CallExpr=operator():3:7
// CHECK7: Identifier: "index" [15:11 - 15:16] DeclRefExpr=index:6:27
// CHECK7: Punctuation: ")" [15:16 - 15:17] CallExpr=operator():3:7
// CHECK7: Punctuation: "+" [15:18 - 15:19] BinaryOperator=
// CHECK7: Identifier: "foo" [15:20 - 15:23] DeclRefExpr=foo:6:18
// CHECK7: Punctuation: "[" [15:23 - 15:24] CallExpr=operator[]:2:7
// CHECK7: Identifier: "index" [15:24 - 15:29] DeclRefExpr=index:6:27
// CHECK7: Punctuation: "]" [15:29 - 15:30] CallExpr=operator[]:2:7
// CHECK7: Punctuation: "]" [15:30 - 15:31] CallExpr=operator[]:2:7
// CHECK7: Punctuation: ";" [15:31 - 15:32] CompoundStmt=

