#include "annotate-tokens-with-default-args.h"

void Foo::m(Foo *f) {}

// RUN: c-index-test -test-annotate-tokens=%s:3:1:4:1 %s | FileCheck %s
// CHECK: Keyword: "void" [3:1 - 3:5] CXXMethod=m:3:11 (Definition)
// CHECK: Identifier: "Foo" [3:6 - 3:9] TypeRef=struct Foo:1:8
// CHECK: Punctuation: "::" [3:9 - 3:11] CXXMethod=m:3:11 (Definition)
// CHECK: Identifier: "m" [3:11 - 3:12] CXXMethod=m:3:11 (Definition)
// CHECK: Punctuation: "(" [3:12 - 3:13] CXXMethod=m:3:11 (Definition)
// CHECK: Identifier: "Foo" [3:13 - 3:16] TypeRef=struct Foo:1:8
// CHECK: Punctuation: "*" [3:17 - 3:18] ParmDecl=f:3:18 (Definition)
// CHECK: Identifier: "f" [3:18 - 3:19] ParmDecl=f:3:18 (Definition)
// CHECK: Punctuation: ")" [3:19 - 3:20] CXXMethod=m:3:11 (Definition)
// CHECK: Punctuation: "{" [3:21 - 3:22] CompoundStmt=
// CHECK: Punctuation: "}" [3:22 - 3:23] CompoundStmt=
