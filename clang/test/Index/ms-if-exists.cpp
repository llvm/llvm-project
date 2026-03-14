template<typename T>
void f(T t) {
  __if_exists(T::foo) {
    { }
    t.foo();
  }

  __if_not_exists(T::bar) {
    int *i = t; // expected-error{{no viable conversion from 'HasFoo' to 'int *'}}
    { }
  }
}

// RUN: c-index-test -test-annotate-tokens=%s:3:1:11:3 -fms-extensions -fno-ms-compatibility -fno-delayed-template-parsing %s | FileCheck %s

// CHECK: Identifier: "T" [3:15 - 3:16] TypeRef=T:1:19
// CHECK: Punctuation: "}" [4:7 - 4:8] CompoundStmt=
// CHECK: Identifier: "t" [5:5 - 5:6] DeclRefExpr=t:2:10
// CHECK: Punctuation: "." [5:6 - 5:7] MemberRefExpr=
// CHECK: Identifier: "foo" [5:7 - 5:10] MemberRefExpr=
// CHECK: Keyword: "int" [9:5 - 9:8] VarDecl=i:9:10 (Definition)
// CHECK: Punctuation: "*" [9:9 - 9:10] VarDecl=i:9:10 (Definition)
// CHECK: Identifier: "i" [9:10 - 9:11] VarDecl=i:9:10 (Definition)
// CHECK: Punctuation: "=" [9:12 - 9:13] VarDecl=i:9:10 (Definition)
