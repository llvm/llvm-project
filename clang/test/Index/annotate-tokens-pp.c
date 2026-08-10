#define NOTHING(X,Y)
#define STILL_NOTHING NOTHING(honk,warble)
#define BAR baz
#define WIBBLE(X, Y) X##Y
NOTHING(more,junk) float WIBBLE(int, float);
int BAR STILL_NOTHING;
#include "foo.h"
#undef BAR

#define REVERSE_MACRO(x,y) y + x
#define TWICE_MACRO(y) y + y

void test_macro_args() {
  int z = 1;
  int t = 2;
  int k = REVERSE_MACRO(t,z);
  int j = TWICE_MACRO(k + k);
  int w = j + j;
}

#define fun_with_macro_bodies(x, y) do { if (x) y } while (0) 

void test() {
  int x = 10;
  fun_with_macro_bodies(x, { int z = x; ++z; });
}

#include "pragma-once.h"
#include "guarded.h"

const char *fname = __FILE__;

#define SOME_MACRO 3

#ifdef SOME_MACRO
#endif

struct A
{
#ifdef SOME_MACRO
  int x;
#endif
};

// RUN: c-index-test -test-annotate-tokens=%s:2:1:44:1 -I%S/Inputs %s | FileCheck %s
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-annotate-tokens=%s:2:1:44:1 -I%S/Inputs %s | FileCheck %s
// CHECK: Punctuation: "#" [2:1 - 2:2] preprocessing directive=
// CHECK: Identifier: "define" [2:2 - 2:8] preprocessing directive=
// CHECK: Identifier: "STILL_NOTHING" [2:9 - 2:22] macro definition=STILL_NOTHING
// CHECK: Identifier: "NOTHING" [2:23 - 2:30] macro expansion=NOTHING:1:9
// CHECK: Punctuation: "(" [2:30 - 2:31] macro definition=STILL_NOTHING
// CHECK: Identifier: "honk" [2:31 - 2:35] macro definition=STILL_NOTHING
// CHECK: Punctuation: "," [2:35 - 2:36] macro definition=STILL_NOTHING
// CHECK: Identifier: "warble" [2:36 - 2:42] macro definition=STILL_NOTHING
// CHECK: Punctuation: ")" [2:42 - 2:43] macro definition=STILL_NOTHING
// CHECK: Punctuation: "#" [3:1 - 3:2] preprocessing directive=
// CHECK: Identifier: "define" [3:2 - 3:8] preprocessing directive=
// CHECK: Identifier: "BAR" [3:9 - 3:12] macro definition=BAR
// CHECK: Identifier: "baz" [3:13 - 3:16] macro definition=BAR
// CHECK: Punctuation: "#" [4:1 - 4:2] preprocessing directive=
// CHECK: Identifier: "define" [4:2 - 4:8] preprocessing directive=
// CHECK: Identifier: "WIBBLE" [4:9 - 4:15] macro definition=WIBBLE
// CHECK: Punctuation: "(" [4:15 - 4:16] macro definition=WIBBLE
// CHECK: Identifier: "X" [4:16 - 4:17] macro definition=WIBBLE
// CHECK: Punctuation: "," [4:17 - 4:18] macro definition=WIBBLE
// CHECK: Identifier: "Y" [4:19 - 4:20] macro definition=WIBBLE
// CHECK: Punctuation: ")" [4:20 - 4:21] macro definition=WIBBLE
// CHECK: Identifier: "X" [4:22 - 4:23] macro definition=WIBBLE
// CHECK: Punctuation: "##" [4:23 - 4:25] macro definition=WIBBLE
// CHECK: Identifier: "Y" [4:25 - 4:26] macro definition=WIBBLE
// CHECK: Identifier: "NOTHING" [5:1 - 5:8] macro expansion=NOTHING:1:9
// CHECK: Punctuation: "(" [5:8 - 5:9]
// CHECK: Identifier: "more" [5:9 - 5:13]
// CHECK: Punctuation: "," [5:13 - 5:14]
// CHECK: Identifier: "junk" [5:14 - 5:18]
// CHECK: Punctuation: ")" [5:18 - 5:19]
// CHECK: Keyword: "float" [5:20 - 5:25]
// CHECK: Identifier: "WIBBLE" [5:26 - 5:32] macro expansion=WIBBLE:4:9
// CHECK: Punctuation: "(" [5:32 - 5:33]
// CHECK: Keyword: "int" [5:33 - 5:36]
// CHECK: Punctuation: "," [5:36 - 5:37]
// CHECK: Keyword: "float" [5:38 - 5:43]
// CHECK: Punctuation: ")" [5:43 - 5:44]
// CHECK: Punctuation: ";" [5:44 - 5:45]
// CHECK: Keyword: "int" [6:1 - 6:4]
// CHECK: Identifier: "BAR" [6:5 - 6:8] macro expansion=BAR:3:9
// CHECK: Identifier: "STILL_NOTHING" [6:9 - 6:22] macro expansion=STILL_NOTHING:2:9
// CHECK: Punctuation: ";" [6:22 - 6:23]
// CHECK: Punctuation: "#" [7:1 - 7:2] inclusion directive=foo.h
// CHECK: Identifier: "include" [7:2 - 7:9] inclusion directive=foo.h
// CHECK: Literal: ""foo.h"" [7:10 - 7:17] inclusion directive=foo.h
// CHECK: Punctuation: "#" [8:1 - 8:2] preprocessing directive=
// CHECK: Identifier: "undef" [8:2 - 8:7] preprocessing directive=
// CHECK: Identifier: "BAR" [8:8 - 8:11] preprocessing directive=
// CHECK: Punctuation: "#" [10:1 - 10:2] preprocessing directive=
// CHECK: Identifier: "define" [10:2 - 10:8] preprocessing directive=
// CHECK: Identifier: "REVERSE_MACRO" [10:9 - 10:22] macro definition=REVERSE_MACRO
// CHECK: Punctuation: "(" [10:22 - 10:23] macro definition=REVERSE_MACRO
// CHECK: Identifier: "x" [10:23 - 10:24] macro definition=REVERSE_MACRO
// CHECK: Punctuation: "," [10:24 - 10:25] macro definition=REVERSE_MACRO
// CHECK: Identifier: "y" [10:25 - 10:26] macro definition=REVERSE_MACRO
// CHECK: Punctuation: ")" [10:26 - 10:27] macro definition=REVERSE_MACRO
// CHECK: Identifier: "y" [10:28 - 10:29] macro definition=REVERSE_MACRO
// CHECK: Punctuation: "+" [10:30 - 10:31] macro definition=REVERSE_MACRO
// CHECK: Identifier: "x" [10:32 - 10:33] macro definition=REVERSE_MACRO
// CHECK: Punctuation: "#" [11:1 - 11:2] preprocessing directive=
// CHECK: Identifier: "define" [11:2 - 11:8] preprocessing directive=
// CHECK: Identifier: "TWICE_MACRO" [11:9 - 11:20] macro definition=TWICE_MACRO
// CHECK: Punctuation: "(" [11:20 - 11:21] macro definition=TWICE_MACRO
// CHECK: Identifier: "y" [11:21 - 11:22] macro definition=TWICE_MACRO
// CHECK: Punctuation: ")" [11:22 - 11:23] macro definition=TWICE_MACRO
// CHECK: Identifier: "y" [11:24 - 11:25] macro definition=TWICE_MACRO
// CHECK: Punctuation: "+" [11:26 - 11:27] macro definition=TWICE_MACRO
// CHECK: Identifier: "y" [11:28 - 11:29] macro definition=TWICE_MACRO
// CHECK: Keyword: "void" [13:1 - 13:5] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Identifier: "test_macro_args" [13:6 - 13:21] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Punctuation: "(" [13:21 - 13:22] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Punctuation: ")" [13:22 - 13:23] FunctionDecl=test_macro_args:13:6 (Definition)
// CHECK: Punctuation: "{" [13:24 - 13:25] CompoundStmt=
// CHECK: Keyword: "int" [14:3 - 14:6] VarDecl=z:14:7 (Definition)
// CHECK: Identifier: "z" [14:7 - 14:8] VarDecl=z:14:7 (Definition)
// CHECK: Punctuation: "=" [14:9 - 14:10] VarDecl=z:14:7 (Definition)
// CHECK: Literal: "1" [14:11 - 14:12] IntegerLiteral=
// CHECK: Punctuation: ";" [14:12 - 14:13] DeclStmt=
// CHECK: Keyword: "int" [15:3 - 15:6] VarDecl=t:15:7 (Definition)
// CHECK: Identifier: "t" [15:7 - 15:8] VarDecl=t:15:7 (Definition)
// CHECK: Punctuation: "=" [15:9 - 15:10] VarDecl=t:15:7 (Definition)
// CHECK: Literal: "2" [15:11 - 15:12] IntegerLiteral=
// CHECK: Punctuation: ";" [15:12 - 15:13] DeclStmt=
// CHECK: Keyword: "int" [16:3 - 16:6] VarDecl=k:16:7 (Definition)
// CHECK: Identifier: "k" [16:7 - 16:8] VarDecl=k:16:7 (Definition)
// CHECK: Punctuation: "=" [16:9 - 16:10] VarDecl=k:16:7 (Definition)
// CHECK: Identifier: "REVERSE_MACRO" [16:11 - 16:24] macro expansion=REVERSE_MACRO:10:9
// CHECK: Punctuation: "(" [16:24 - 16:25]
// CHECK: Identifier: "t" [16:25 - 16:26] DeclRefExpr=t:15:7
// CHECK: Punctuation: "," [16:26 - 16:27]
// CHECK: Identifier: "z" [16:27 - 16:28] DeclRefExpr=z:14:7
// FIXME: The token below should really be annotated as "macro expansion=REVERSE_MACRO:10:9"
// CHECK: Punctuation: ")" [16:28 - 16:29] DeclStmt=
// CHECK: Punctuation: ";" [16:29 - 16:30] DeclStmt=
// CHECK: Keyword: "int" [17:3 - 17:6] VarDecl=j:17:7 (Definition)
// CHECK: Identifier: "j" [17:7 - 17:8] VarDecl=j:17:7 (Definition)
// CHECK: Punctuation: "=" [17:9 - 17:10] VarDecl=j:17:7 (Definition)
// CHECK: Identifier: "TWICE_MACRO" [17:11 - 17:22] macro expansion=TWICE_MACRO:11:9
// CHECK: Punctuation: "(" [17:22 - 17:23]
// CHECK: Identifier: "k" [17:23 - 17:24] DeclRefExpr=k:16:7
// CHECK: Punctuation: "+" [17:25 - 17:26] BinaryOperator=
// CHECK: Identifier: "k" [17:27 - 17:28] DeclRefExpr=k:16:7
// FIXME: The token below should really be annotated as "macro expansion=TWICE_MACRO:11:9"
// CHECK: Punctuation: ")" [17:28 - 17:29] DeclStmt=
// CHECK: Punctuation: ";" [17:29 - 17:30] DeclStmt=
// CHECK: Keyword: "int" [18:3 - 18:6] VarDecl=w:18:7 (Definition)
// CHECK: Identifier: "w" [18:7 - 18:8] VarDecl=w:18:7 (Definition)
// CHECK: Punctuation: "=" [18:9 - 18:10] VarDecl=w:18:7 (Definition)
// CHECK: Identifier: "j" [18:11 - 18:12] DeclRefExpr=j:17:7
// CHECK: Punctuation: "+" [18:13 - 18:14] BinaryOperator=
// CHECK: Identifier: "j" [18:15 - 18:16] DeclRefExpr=j:17:7
// CHECK: Punctuation: ";" [18:16 - 18:17] DeclStmt=
// CHECK: Punctuation: "}" [19:1 - 19:2] CompoundStmt=
// CHECK: Punctuation: "#" [21:1 - 21:2] preprocessing directive=
// CHECK: Identifier: "define" [21:2 - 21:8] preprocessing directive=
// CHECK: Identifier: "fun_with_macro_bodies" [21:9 - 21:30] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: "(" [21:30 - 21:31] macro definition=fun_with_macro_bodies
// CHECK: Identifier: "x" [21:31 - 21:32] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: "," [21:32 - 21:33] macro definition=fun_with_macro_bodies
// CHECK: Identifier: "y" [21:34 - 21:35] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: ")" [21:35 - 21:36] macro definition=fun_with_macro_bodies
// CHECK: Keyword: "do" [21:37 - 21:39] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: "{" [21:40 - 21:41] macro definition=fun_with_macro_bodies
// CHECK: Keyword: "if" [21:42 - 21:44] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: "(" [21:45 - 21:46] macro definition=fun_with_macro_bodies
// CHECK: Identifier: "x" [21:46 - 21:47] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: ")" [21:47 - 21:48] macro definition=fun_with_macro_bodies
// CHECK: Identifier: "y" [21:49 - 21:50] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: "}" [21:51 - 21:52] macro definition=fun_with_macro_bodies
// CHECK: Keyword: "while" [21:53 - 21:58] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: "(" [21:59 - 21:60] macro definition=fun_with_macro_bodies
// CHECK: Literal: "0" [21:60 - 21:61] macro definition=fun_with_macro_bodies
// CHECK: Punctuation: ")" [21:61 - 21:62] macro definition=fun_with_macro_bodies
// CHECK: Keyword: "void" [23:1 - 23:5] FunctionDecl=test:23:6 (Definition)
// CHECK: Identifier: "test" [23:6 - 23:10] FunctionDecl=test:23:6 (Definition)
// CHECK: Punctuation: "(" [23:10 - 23:11] FunctionDecl=test:23:6 (Definition)
// CHECK: Punctuation: ")" [23:11 - 23:12] FunctionDecl=test:23:6 (Definition)
// CHECK: Punctuation: "{" [23:13 - 23:14] CompoundStmt=
// CHECK: Keyword: "int" [24:3 - 24:6] VarDecl=x:24:7 (Definition)
// CHECK: Identifier: "x" [24:7 - 24:8] VarDecl=x:24:7 (Definition)
// CHECK: Punctuation: "=" [24:9 - 24:10] VarDecl=x:24:7 (Definition)
// CHECK: Literal: "10" [24:11 - 24:13] IntegerLiteral=
// CHECK: Punctuation: ";" [24:13 - 24:14] DeclStmt=
// CHECK: Identifier: "fun_with_macro_bodies" [25:3 - 25:24] macro expansion=fun_with_macro_bodies:21:9
// CHECK: Punctuation: "(" [25:24 - 25:25] CompoundStmt=
// CHECK: Identifier: "x" [25:25 - 25:26] DeclRefExpr=x:24:7
// CHECK: Punctuation: "," [25:26 - 25:27]
// CHECK: Punctuation: "{" [25:28 - 25:29] CompoundStmt=
// CHECK: Keyword: "int" [25:30 - 25:33] VarDecl=z:25:34 (Definition)
// CHECK: Identifier: "z" [25:34 - 25:35] VarDecl=z:25:34 (Definition)
// CHECK: Punctuation: "=" [25:36 - 25:37] VarDecl=z:25:34 (Definition)
// CHECK: Identifier: "x" [25:38 - 25:39] DeclRefExpr=x:24:7
// CHECK: Punctuation: ";" [25:39 - 25:40] DeclStmt=
// CHECK: Punctuation: "++" [25:41 - 25:43] UnaryOperator=
// CHECK: Identifier: "z" [25:43 - 25:44] DeclRefExpr=z:25:3
// CHECK: Punctuation: ";" [25:44 - 25:45] CompoundStmt=
// CHECK: Punctuation: "}" [25:46 - 25:47] CompoundStmt=
// FIXME: The token below should really be annotated as "macro expansion=fun_with_macro_bodies:21:9"
// CHECK: Punctuation: ")" [25:47 - 25:48] CompoundStmt=
// CHECK: Punctuation: ";" [25:48 - 25:49] CompoundStmt=
// CHECK: Punctuation: "}" [26:1 - 26:2] CompoundStmt=
// CHECK: {{28:1.*inclusion directive=pragma-once.h.*multi-include guarded}}
// CHECK: {{29:1.*inclusion directive=guarded.h.*multi-include guarded}}
// CHECK: Identifier: "__FILE__" [31:21 - 31:29] macro expansion=__FILE__
// CHECK: Punctuation: "#" [35:1 - 35:2] preprocessing directive=
// CHECK: Identifier: "ifdef" [35:2 - 35:7] preprocessing directive=
// CHECK: Identifier: "SOME_MACRO" [35:8 - 35:18] macro expansion=SOME_MACRO:33:9
// CHECK: Punctuation: "#" [36:1 - 36:2] preprocessing directive=
// CHECK: Identifier: "endif" [36:2 - 36:7] preprocessing directive=
// CHECK: Keyword: "struct" [38:1 - 38:7] StructDecl=A:38:8 (Definition)
// CHECK: Identifier: "A" [38:8 - 38:9] StructDecl=A:38:8 (Definition)
// CHECK: Punctuation: "{" [39:1 - 39:2] StructDecl=A:38:8 (Definition)
// CHECK: Punctuation: "#" [40:1 - 40:2] preprocessing directive=
// CHECK: Identifier: "ifdef" [40:2 - 40:7] preprocessing directive=
// CHECK: Identifier: "SOME_MACRO" [40:8 - 40:18] macro expansion=SOME_MACRO:33:9
// CHECK: Keyword: "int" [41:3 - 41:6] FieldDecl=x:41:7 (Definition)
// CHECK: Identifier: "x" [41:7 - 41:8] FieldDecl=x:41:7 (Definition)
// CHECK: Punctuation: ";" [41:8 - 41:9] StructDecl=A:38:8 (Definition)
// CHECK: Punctuation: "#" [42:1 - 42:2] preprocessing directive=
// CHECK: Identifier: "endif" [42:2 - 42:7] preprocessing directive=
// CHECK: Punctuation: "}" [43:1 - 43:2] StructDecl=A:38:8 (Definition)
