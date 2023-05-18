struct bonk { };
void test(bonk X) {
    X = X;
    __is_base_of(bonk, bonk);
}

struct X {
  X operator++();
  X operator++(int);
};
void test2(X x) {
  ++(x);
  (x)++;
}

struct S1 { void f(); };
struct S2 { S1 *operator->(); };
void test3(S2 s2) {
  s2->f();
  X foo;
}

template <bool (*tfn)(X*)>
struct TS {
  void foo();
};

template <bool (*tfn)(X*)>
void TS<tfn>::foo() {}

void test4() {
  if (int p = 0)
    return;
}

class C {
  ~C();
};

auto test5(X) -> X;

// RUN: c-index-test -test-annotate-tokens=%s:1:1:41:1 %s -std=c++14 -fno-delayed-template-parsing | FileCheck %s
// CHECK: Keyword: "struct" [1:1 - 1:7] StructDecl=bonk:1:8 (Definition)
// CHECK: Identifier: "bonk" [1:8 - 1:12] StructDecl=bonk:1:8 (Definition)
// CHECK: Punctuation: "{" [1:13 - 1:14] StructDecl=bonk:1:8 (Definition)
// CHECK: Punctuation: "}" [1:15 - 1:16] StructDecl=bonk:1:8 (Definition)
// CHECK: Punctuation: ";" [1:16 - 1:17]
// CHECK: Keyword: "void" [2:1 - 2:5] FunctionDecl=test:2:6 (Definition)
// CHECK: Identifier: "test" [2:6 - 2:10] FunctionDecl=test:2:6 (Definition)
// CHECK: Punctuation: "(" [2:10 - 2:11] FunctionDecl=test:2:6 (Definition)
// CHECK: Identifier: "bonk" [2:11 - 2:15] TypeRef=struct bonk:1:8
// CHECK: Identifier: "X" [2:16 - 2:17] ParmDecl=X:2:16 (Definition)
// CHECK: Punctuation: ")" [2:17 - 2:18] FunctionDecl=test:2:6 (Definition)
// CHECK: Punctuation: "{" [2:19 - 2:20] CompoundStmt=
// CHECK: Identifier: "X" [3:5 - 3:6] DeclRefExpr=X:2:16
// CHECK: Punctuation: "=" [3:7 - 3:8] DeclRefExpr=operator=:1:8
// CHECK: Identifier: "X" [3:9 - 3:10] DeclRefExpr=X:2:16
// CHECK: Punctuation: ";" [3:10 - 3:11] CompoundStmt=
// CHECK: Keyword: "__is_base_of" [4:5 - 4:17] UnexposedExpr=
// CHECK: Punctuation: "(" [4:17 - 4:18] UnexposedExpr=
// CHECK: Identifier: "bonk" [4:18 - 4:22] TypeRef=struct bonk:1:8
// CHECK: Punctuation: "," [4:22 - 4:23] UnexposedExpr=
// CHECK: Identifier: "bonk" [4:24 - 4:28] TypeRef=struct bonk:1:8
// CHECK: Punctuation: ")" [4:28 - 4:29] UnexposedExpr=
// CHECK: Punctuation: ";" [4:29 - 4:30] CompoundStmt=
// CHECK: Punctuation: "}" [5:1 - 5:2] CompoundStmt=
// CHECK: Keyword: "struct" [7:1 - 7:7] StructDecl=X:7:8 (Definition)
// CHECK: Identifier: "X" [7:8 - 7:9] StructDecl=X:7:8 (Definition)
// CHECK: Punctuation: "{" [7:10 - 7:11] StructDecl=X:7:8 (Definition)
// CHECK: Identifier: "X" [8:3 - 8:4] TypeRef=struct X:7:8
// CHECK: Keyword: "operator" [8:5 - 8:13] CXXMethod=operator++:8:5
// CHECK: Punctuation: "++" [8:13 - 8:15] CXXMethod=operator++:8:5
// CHECK: Punctuation: "(" [8:15 - 8:16] CXXMethod=operator++:8:5
// CHECK: Punctuation: ")" [8:16 - 8:17] CXXMethod=operator++:8:5
// CHECK: Punctuation: ";" [8:17 - 8:18] StructDecl=X:7:8 (Definition)
// CHECK: Identifier: "X" [9:3 - 9:4] TypeRef=struct X:7:8
// CHECK: Keyword: "operator" [9:5 - 9:13] CXXMethod=operator++:9:5
// CHECK: Punctuation: "++" [9:13 - 9:15] CXXMethod=operator++:9:5
// CHECK: Punctuation: "(" [9:15 - 9:16] CXXMethod=operator++:9:5
// CHECK: Keyword: "int" [9:16 - 9:19] ParmDecl=:9:19 (Definition
// CHECK: Punctuation: ")" [9:19 - 9:20] CXXMethod=operator++:9:5
// CHECK: Punctuation: ";" [9:20 - 9:21] StructDecl=X:7:8 (Definition)
// CHECK: Punctuation: "}" [10:1 - 10:2] StructDecl=X:7:8 (Definition)
// CHECK: Punctuation: ";" [10:2 - 10:3]
// CHECK: Keyword: "void" [11:1 - 11:5] FunctionDecl=test2:11:6 (Definition)
// CHECK: Identifier: "test2" [11:6 - 11:11] FunctionDecl=test2:11:6 (Definition)
// CHECK: Punctuation: "(" [11:11 - 11:12] FunctionDecl=test2:11:6 (Definition)
// CHECK: Identifier: "X" [11:12 - 11:13] TypeRef=struct X:7:8
// CHECK: Identifier: "x" [11:14 - 11:15] ParmDecl=x:11:14 (Definition)
// CHECK: Punctuation: ")" [11:15 - 11:16] FunctionDecl=test2:11:6 (Definition)
// CHECK: Punctuation: "{" [11:17 - 11:18] CompoundStmt=
// CHECK: Punctuation: "++" [12:3 - 12:5] CallExpr=operator++:8:5
// CHECK: Punctuation: "(" [12:5 - 12:6] ParenExpr=
// CHECK: Identifier: "x" [12:6 - 12:7] DeclRefExpr=x:11:14
// CHECK: Punctuation: ")" [12:7 - 12:8] ParenExpr=
// CHECK: Punctuation: ";" [12:8 - 12:9] CompoundStmt=
// CHECK: Punctuation: "(" [13:3 - 13:4] ParenExpr=
// CHECK: Identifier: "x" [13:4 - 13:5] DeclRefExpr=x:11:14
// CHECK: Punctuation: ")" [13:5 - 13:6] ParenExpr=
// CHECK: Punctuation: "++" [13:6 - 13:8] DeclRefExpr=operator++:9:5
// CHECK: Punctuation: ";" [13:8 - 13:9] CompoundStmt=
// CHECK: Punctuation: "}" [14:1 - 14:2] CompoundStmt=
// CHECK: Keyword: "struct" [16:1 - 16:7] StructDecl=S1:16:8 (Definition)
// CHECK: Identifier: "S1" [16:8 - 16:10] StructDecl=S1:16:8 (Definition)
// CHECK: Punctuation: "{" [16:11 - 16:12] StructDecl=S1:16:8 (Definition)
// CHECK: Keyword: "void" [16:13 - 16:17] CXXMethod=f:16:18
// CHECK: Identifier: "f" [16:18 - 16:19] CXXMethod=f:16:18
// CHECK: Punctuation: "(" [16:19 - 16:20] CXXMethod=f:16:18
// CHECK: Punctuation: ")" [16:20 - 16:21] CXXMethod=f:16:18
// CHECK: Punctuation: ";" [16:21 - 16:22] StructDecl=S1:16:8 (Definition)
// CHECK: Punctuation: "}" [16:23 - 16:24] StructDecl=S1:16:8 (Definition)
// CHECK: Punctuation: ";" [16:24 - 16:25]
// CHECK: Keyword: "struct" [17:1 - 17:7] StructDecl=S2:17:8 (Definition)
// CHECK: Identifier: "S2" [17:8 - 17:10] StructDecl=S2:17:8 (Definition)
// CHECK: Punctuation: "{" [17:11 - 17:12] StructDecl=S2:17:8 (Definition)
// CHECK: Identifier: "S1" [17:13 - 17:15] TypeRef=struct S1:16:8
// CHECK: Punctuation: "*" [17:16 - 17:17] CXXMethod=operator->:17:17
// CHECK: Keyword: "operator" [17:17 - 17:25] CXXMethod=operator->:17:17
// CHECK: Punctuation: "->" [17:25 - 17:27] CXXMethod=operator->:17:17
// CHECK: Punctuation: "(" [17:27 - 17:28] CXXMethod=operator->:17:17
// CHECK: Punctuation: ")" [17:28 - 17:29] CXXMethod=operator->:17:17
// CHECK: Punctuation: ";" [17:29 - 17:30] StructDecl=S2:17:8 (Definition)
// CHECK: Punctuation: "}" [17:31 - 17:32] StructDecl=S2:17:8 (Definition)
// CHECK: Punctuation: ";" [17:32 - 17:33]
// CHECK: Keyword: "void" [18:1 - 18:5] FunctionDecl=test3:18:6 (Definition)
// CHECK: Identifier: "test3" [18:6 - 18:11] FunctionDecl=test3:18:6 (Definition)
// CHECK: Punctuation: "(" [18:11 - 18:12] FunctionDecl=test3:18:6 (Definition)
// CHECK: Identifier: "S2" [18:12 - 18:14] TypeRef=struct S2:17:8
// CHECK: Identifier: "s2" [18:15 - 18:17] ParmDecl=s2:18:15 (Definition)
// CHECK: Punctuation: ")" [18:17 - 18:18] FunctionDecl=test3:18:6 (Definition)
// CHECK: Punctuation: "{" [18:19 - 18:20] CompoundStmt=
// CHECK: Identifier: "s2" [19:3 - 19:5] DeclRefExpr=s2:18:15
// CHECK: Punctuation: "->" [19:5 - 19:7] DeclRefExpr=operator->:17:17
// CHECK: Identifier: "f" [19:7 - 19:8] MemberRefExpr=f:16:18
// CHECK: Punctuation: "(" [19:8 - 19:9] CallExpr=f:16:18
// CHECK: Punctuation: ")" [19:9 - 19:10] CallExpr=f:16:18
// CHECK: Punctuation: ";" [19:10 - 19:11] CompoundStmt=
// CHECK: Identifier: "X" [20:3 - 20:4] TypeRef=struct X:7:8
// CHECK: Identifier: "foo" [20:5 - 20:8] VarDecl=foo:20:5 (Definition)
// CHECK: Punctuation: ";" [20:8 - 20:9] DeclStmt=
// CHECK: Punctuation: "}" [21:1 - 21:2] CompoundStmt=
// CHECK: Keyword: "template" [23:1 - 23:9] ClassTemplate=TS:24:8 (Definition)
// CHECK: Punctuation: "<" [23:10 - 23:11] ClassTemplate=TS:24:8 (Definition)
// CHECK: Keyword: "bool" [23:11 - 23:15] NonTypeTemplateParameter=tfn:23:18 (Definition)
// CHECK: Punctuation: "(" [23:16 - 23:17] NonTypeTemplateParameter=tfn:23:18 (Definition)
// CHECK: Punctuation: "*" [23:17 - 23:18] NonTypeTemplateParameter=tfn:23:18 (Definition)
// CHECK: Identifier: "tfn" [23:18 - 23:21] NonTypeTemplateParameter=tfn:23:18 (Definition)
// CHECK: Punctuation: ")" [23:21 - 23:22] NonTypeTemplateParameter=tfn:23:18 (Definition)
// CHECK: Punctuation: "(" [23:22 - 23:23] NonTypeTemplateParameter=tfn:23:18 (Definition)
// CHECK: Identifier: "X" [23:23 - 23:24] TypeRef=struct X:7:8
// CHECK: Punctuation: "*" [23:24 - 23:25] ParmDecl=:23:25 (Definition)
// CHECK: Punctuation: ")" [23:25 - 23:26] NonTypeTemplateParameter=tfn:23:18 (Definition)
// CHECK: Punctuation: ">" [23:26 - 23:27] ClassTemplate=TS:24:8 (Definition)
// CHECK: Keyword: "struct" [24:1 - 24:7] ClassTemplate=TS:24:8 (Definition)
// CHECK: Identifier: "TS" [24:8 - 24:10] ClassTemplate=TS:24:8 (Definition)
// CHECK: Punctuation: "{" [24:11 - 24:12] ClassTemplate=TS:24:8 (Definition)
// CHECK: Keyword: "void" [25:3 - 25:7] CXXMethod=foo:25:8
// CHECK: Identifier: "foo" [25:8 - 25:11] CXXMethod=foo:25:8
// CHECK: Punctuation: "(" [25:11 - 25:12] CXXMethod=foo:25:8
// CHECK: Punctuation: ")" [25:12 - 25:13] CXXMethod=foo:25:8
// CHECK: Punctuation: ";" [25:13 - 25:14] ClassTemplate=TS:24:8 (Definition)
// CHECK: Punctuation: "}" [26:1 - 26:2] ClassTemplate=TS:24:8 (Definition)
// CHECK: Punctuation: ";" [26:2 - 26:3]
// CHECK: Keyword: "template" [28:1 - 28:9] CXXMethod=foo:29:15 (Definition)
// CHECK: Punctuation: "<" [28:10 - 28:11] CXXMethod=foo:29:15 (Definition)
// CHECK: Keyword: "bool" [28:11 - 28:15] NonTypeTemplateParameter=tfn:28:18 (Definition)
// CHECK: Punctuation: "(" [28:16 - 28:17] NonTypeTemplateParameter=tfn:28:18 (Definition)
// CHECK: Punctuation: "*" [28:17 - 28:18] NonTypeTemplateParameter=tfn:28:18 (Definition)
// CHECK: Identifier: "tfn" [28:18 - 28:21] NonTypeTemplateParameter=tfn:28:18 (Definition)
// CHECK: Punctuation: ")" [28:21 - 28:22] NonTypeTemplateParameter=tfn:28:18 (Definition)
// CHECK: Punctuation: "(" [28:22 - 28:23] NonTypeTemplateParameter=tfn:28:18 (Definition)
// CHECK: Identifier: "X" [28:23 - 28:24] TypeRef=struct X:7:8
// CHECK: Punctuation: "*" [28:24 - 28:25] ParmDecl=:28:25 (Definition)
// CHECK: Punctuation: ")" [28:25 - 28:26] NonTypeTemplateParameter=tfn:28:18 (Definition)
// CHECK: Punctuation: ">" [28:26 - 28:27] CXXMethod=foo:29:15 (Definition)
// CHECK: Keyword: "void" [29:1 - 29:5] CXXMethod=foo:29:15 (Definition)
// CHECK: Identifier: "TS" [29:6 - 29:8] TemplateRef=TS:24:8
// CHECK: Punctuation: "<" [29:8 - 29:9] CXXMethod=foo:29:15 (Definition)
// CHECK: Identifier: "tfn" [29:9 - 29:12] DeclRefExpr=tfn:28:18
// CHECK: Punctuation: ">" [29:12 - 29:13] CXXMethod=foo:29:15 (Definition)
// CHECK: Punctuation: "::" [29:13 - 29:15] CXXMethod=foo:29:15 (Definition)
// CHECK: Identifier: "foo" [29:15 - 29:18] CXXMethod=foo:29:15 (Definition)
// CHECK: Punctuation: "(" [29:18 - 29:19] CXXMethod=foo:29:15 (Definition)
// CHECK: Punctuation: ")" [29:19 - 29:20] CXXMethod=foo:29:15 (Definition)
// CHECK: Punctuation: "{" [29:21 - 29:22] CompoundStmt=
// CHECK: Punctuation: "}" [29:22 - 29:23] CompoundStmt=
// CHECK: Punctuation: "~" [37:3 - 37:4] CXXDestructor=~C:37:3
// CHECK: Identifier: "C" [37:4 - 37:5] CXXDestructor=~C:37:3
// CHECK: Keyword: "auto" [40:1 - 40:5] FunctionDecl=test5:40:6
// CHECK: Identifier: "test5" [40:6 - 40:11] FunctionDecl=test5:40:6
// CHECK: Punctuation: "(" [40:11 - 40:12] FunctionDecl=test5:40:6
// CHECK: Identifier: "X" [40:12 - 40:13] TypeRef=struct X:7:8
// CHECK: Punctuation: ")" [40:13 - 40:14] FunctionDecl=test5:40:6
// CHECK: Punctuation: "->" [40:15 - 40:17] FunctionDecl=test5:40:6
// CHECK: Identifier: "X" [40:18 - 40:19] TypeRef=struct X:7:8
// CHECK: Punctuation: ";" [40:19 - 40:20]

// RUN: env LIBCLANG_DISABLE_CRASH_RECOVERY=1 c-index-test -test-annotate-tokens=%s:32:1:32:13 %s | FileCheck %s -check-prefix=CHECK2
// CHECK2: Keyword: "if" [32:3 - 32:5] IfStmt=
// CHECK2: Punctuation: "(" [32:6 - 32:7] IfStmt=
// CHECK2: Keyword: "int" [32:7 - 32:10] VarDecl=p:32:11 (Definition)
// CHECK2: Identifier: "p" [32:11 - 32:12] VarDecl=p:32:11 (Definition)
// CHECK2: Punctuation: "=" [32:13 - 32:14] VarDecl=p:32:11 (Definition)
