typedef int T;
struct X { int a, b; };
void f(void *ptr) {
  T* t_ptr = (T *)ptr;
  (void)sizeof(T);
  /* A comment */
  struct X x = (struct X){1, 2};
  void *xx = ptr ? : &x;
  const char * hello = "Hello";
}
enum Color { Red, Green, Blue };
typedef int Int;
enum Color g(int i, ...) {
  __builtin_va_list va;
  (void)__builtin_va_arg(va, Int);
  (void)__builtin_types_compatible_p(Int, Int);

  struct X x = { 0, 0 };
  do {
    x.a++;
  } while (x.a < 10);
  
  enum Color c;
  switch (c) {
  case Red:
    return Green;

  case Green:
    return Blue;

  case Blue:
    return Red;
  }
}

__attribute__((unavailable)) Int __attribute__((unavailable)) test() __attribute__((unavailable));

#define HEADER() \
    int x; \
    int y; \
    int z

#define TYPE_INST(name, ...) \
    static const struct { \
        HEADER(); \
    } name = { \
        __VA_ARGS__ \
    }

void func1(void);

TYPE_INST(Foo,
    .x = 0,
    .y = 1,
    .z = 2,
);

void func2(void);

typedef union {
  struct {
    int field : 16;
  };
} r_t;

void test() {
  r_t reg;
  reg.field = 1;
}

// RUN: c-index-test -test-annotate-tokens=%s:4:1:37:1 %s | FileCheck %s
// CHECK: Identifier: "T" [4:3 - 4:4] TypeRef=T:1:13
// CHECK: Punctuation: "*" [4:4 - 4:5] VarDecl=t_ptr:4:6 (Definition)
// CHECK: Identifier: "t_ptr" [4:6 - 4:11] VarDecl=t_ptr:4:6 (Definition)
// CHECK: Punctuation: "=" [4:12 - 4:13] VarDecl=t_ptr:4:6 (Definition)
// CHECK: Punctuation: "(" [4:14 - 4:15] CStyleCastExpr=
// CHECK: Identifier: "T" [4:15 - 4:16] TypeRef=T:1:13
// CHECK: Identifier: "ptr" [4:19 - 4:22] DeclRefExpr=ptr:3:14
// CHECK: Punctuation: ";" [4:22 - 4:23] DeclStmt=
// CHECK: Punctuation: "(" [5:3 - 5:4] CStyleCastExpr=
// CHECK: Keyword: "void" [5:4 - 5:8] CStyleCastExpr=
// CHECK: Punctuation: ")" [5:8 - 5:9] CStyleCastExpr=
// CHECK: Keyword: "sizeof" [5:9 - 5:15] UnaryExpr=
// CHECK: Punctuation: "(" [5:15 - 5:16] UnaryExpr=
// CHECK: Identifier: "T" [5:16 - 5:17] TypeRef=T:1:13
// CHECK: Punctuation: ")" [5:17 - 5:18] UnaryExpr=
// CHECK: Punctuation: ";" [5:18 - 5:19] CompoundStmt=
// CHECK: Keyword: "struct" [7:3 - 7:9] VarDecl=x:7:12 (Definition)
// CHECK: Identifier: "X" [7:10 - 7:11] TypeRef=struct X:2:8
// CHECK: Identifier: "x" [7:12 - 7:13] VarDecl=x:7:12 (Definition)
// CHECK: Punctuation: "=" [7:14 - 7:15] VarDecl=x:7:12 (Definition)
// CHECK: Punctuation: "(" [7:16 - 7:17] CompoundLiteralExpr=
// CHECK: Keyword: "struct" [7:17 - 7:23] CompoundLiteralExpr=
// CHECK: Identifier: "X" [7:24 - 7:25] TypeRef=struct X:2:8
// CHECK: Punctuation: ")" [7:25 - 7:26] CompoundLiteralExpr=
// CHECK: Punctuation: "{" [7:26 - 7:27] InitListExpr=
// CHECK: Literal: "1" [7:27 - 7:28] IntegerLiteral=
// CHECK: Punctuation: "," [7:28 - 7:29] InitListExpr=
// CHECK: Literal: "2" [7:30 - 7:31] IntegerLiteral=
// CHECK: Punctuation: "}" [7:31 - 7:32] InitListExpr=
// CHECK: Punctuation: ";" [7:32 - 7:33] DeclStmt=
// CHECK: Keyword: "void" [8:3 - 8:7] VarDecl=xx:8:9 (Definition)
// CHECK: Punctuation: "*" [8:8 - 8:9] VarDecl=xx:8:9 (Definition)
// CHECK: Identifier: "xx" [8:9 - 8:11] VarDecl=xx:8:9 (Definition)
// CHECK: Punctuation: "=" [8:12 - 8:13] VarDecl=xx:8:9 (Definition)
// CHECK: Identifier: "ptr" [8:14 - 8:17] DeclRefExpr=ptr:3:14
// CHECK: Punctuation: "?" [8:18 - 8:19] UnexposedExpr=
// CHECK: Punctuation: ":" [8:20 - 8:21] UnexposedExpr=
// CHECK: Punctuation: "&" [8:22 - 8:23] UnaryOperator=
// CHECK: Identifier: "x" [8:23 - 8:24] DeclRefExpr=x:7:12
// CHECK: Punctuation: ";" [8:24 - 8:25] DeclStmt=
// CHECK: Keyword: "const" [9:3 - 9:8] VarDecl=hello:9:16 (Definition)
// CHECK: Keyword: "char" [9:9 - 9:13] VarDecl=hello:9:16 (Definition)
// CHECK: Punctuation: "*" [9:14 - 9:15] VarDecl=hello:9:16 (Definition)
// CHECK: Identifier: "hello" [9:16 - 9:21] VarDecl=hello:9:16 (Definition)
// CHECK: Punctuation: "=" [9:22 - 9:23] VarDecl=hello:9:16 (Definition)
// CHECK: Literal: ""Hello"" [9:24 - 9:31] StringLiteral=
// CHECK: Punctuation: ";" [9:31 - 9:32] DeclStmt=
// CHECK: Punctuation: "}" [10:1 - 10:2] CompoundStmt=
// CHECK: Keyword: "__builtin_va_arg" [15:9 - 15:25] UnexposedExpr=
// CHECK: Identifier: "Int" [15:30 - 15:33] TypeRef=Int:12:13
// CHECK: Keyword: "__builtin_types_compatible_p" [16:9 - 16:37] UnexposedExpr=
// CHECK: Identifier: "Int" [16:38 - 16:41] TypeRef=Int:12:13
// CHECK: Punctuation: "," [16:41 - 16:42] UnexposedExpr=
// CHECK: Identifier: "Int" [16:43 - 16:46] TypeRef=Int:12:13
// CHECK: Keyword: "struct" [18:3 - 18:9] VarDecl=x:18:12 (Definition)
// CHECK: Identifier: "X" [18:10 - 18:11] TypeRef=struct X:2:8
// CHECK: Identifier: "x" [18:12 - 18:13] VarDecl=x:18:12 (Definition)
// CHECK: Keyword: "do" [19:3 - 19:5] DoStmt=
// CHECK: Identifier: "x" [20:5 - 20:6] DeclRefExpr=x:18:12
// CHECK: Punctuation: "." [20:6 - 20:7] MemberRefExpr=a:2:16
// CHECK: Identifier: "a" [20:7 - 20:8] MemberRefExpr=a:2:16
// CHECK: Punctuation: "++" [20:8 - 20:10] UnaryOperator=
// CHECK: Punctuation: ";" [20:10 - 20:11] CompoundStmt=
// CHECK: Punctuation: "}" [21:3 - 21:4] CompoundStmt=
// CHECK: Keyword: "while" [21:5 - 21:10] DoStmt=
// CHECK: Punctuation: "(" [21:11 - 21:12] DoStmt=
// CHECK: Identifier: "x" [21:12 - 21:13] DeclRefExpr=x:18:12
// CHECK: Punctuation: "." [21:13 - 21:14] MemberRefExpr=a:2:16
// CHECK: Identifier: "a" [21:14 - 21:15] MemberRefExpr=a:2:16

// CHECK: Keyword: "enum" [23:3 - 23:7] VarDecl=c:23:14 (Definition)
// CHECK: Identifier: "Color" [23:8 - 23:13] TypeRef=enum Color:11:6
// CHECK: Identifier: "c" [23:14 - 23:15] VarDecl=c:23:14 (Definition)
// CHECK: Punctuation: ";" [23:15 - 23:16] DeclStmt=
// CHECK: Keyword: "switch" [24:3 - 24:9] SwitchStmt=
// CHECK: Punctuation: "(" [24:10 - 24:11] SwitchStmt=
// CHECK: Identifier: "c" [24:11 - 24:12] DeclRefExpr=c:23:14
// CHECK: Punctuation: ")" [24:12 - 24:13] SwitchStmt=
// CHECK: Punctuation: "{" [24:14 - 24:15] CompoundStmt=
// CHECK: Keyword: "case" [25:3 - 25:7] CaseStmt=
// CHECK: Identifier: "Red" [25:8 - 25:11] DeclRefExpr=Red:11:14
// CHECK: Punctuation: ":" [25:11 - 25:12] CaseStmt=
// CHECK: Keyword: "return" [26:5 - 26:11] ReturnStmt=
// CHECK: Identifier: "Green" [26:12 - 26:17] DeclRefExpr=Green:11:19
// CHECK: Punctuation: ";" [26:17 - 26:18] CompoundStmt=
// CHECK: Keyword: "case" [28:3 - 28:7] CaseStmt=
// CHECK: Identifier: "Green" [28:8 - 28:13] DeclRefExpr=Green:11:19
// CHECK: Punctuation: ":" [28:13 - 28:14] CaseStmt=
// CHECK: Keyword: "return" [29:5 - 29:11] ReturnStmt=
// CHECK: Identifier: "Blue" [29:12 - 29:16] DeclRefExpr=Blue:11:26
// CHECK: Punctuation: ";" [29:16 - 29:17] CompoundStmt=
// CHECK: Keyword: "case" [31:3 - 31:7] CaseStmt=
// CHECK: Identifier: "Blue" [31:8 - 31:12] DeclRefExpr=Blue:11:26
// CHECK: Punctuation: ":" [31:12 - 31:13] CaseStmt=
// CHECK: Keyword: "return" [32:5 - 32:11] ReturnStmt=
// CHECK: Identifier: "Red" [32:12 - 32:15] DeclRefExpr=Red:11:14
// CHECK: Punctuation: ";" [32:15 - 32:16] CompoundStmt=

// CHECK: Keyword: "__attribute__" [36:1 - 36:14] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: "(" [36:14 - 36:15] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: "(" [36:15 - 36:16] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Identifier: "unavailable" [36:16 - 36:27] UnexposedAttr=
// CHECK: Punctuation: ")" [36:27 - 36:28] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: ")" [36:28 - 36:29] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Identifier: "Int" [36:30 - 36:33] TypeRef=Int:12:13
// CHECK: Keyword: "__attribute__" [36:34 - 36:47] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: "(" [36:47 - 36:48] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: "(" [36:48 - 36:49] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Identifier: "unavailable" [36:49 - 36:60] UnexposedAttr=
// CHECK: Punctuation: ")" [36:60 - 36:61] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: ")" [36:61 - 36:62] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Identifier: "test" [36:63 - 36:67] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: "(" [36:67 - 36:68] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: ")" [36:68 - 36:69] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Keyword: "__attribute__" [36:70 - 36:83] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: "(" [36:83 - 36:84] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: "(" [36:84 - 36:85] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Identifier: "unavailable" [36:85 - 36:96] UnexposedAttr=
// CHECK: Punctuation: ")" [36:96 - 36:97] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: ")" [36:97 - 36:98] FunctionDecl=test:36:63 (unavailable)  (always unavailable: "")
// CHECK: Punctuation: ";" [36:98 - 36:99]

// RUN: c-index-test -test-annotate-tokens=%s:4:1:165:32 %s | FileCheck %s
// RUN: c-index-test -test-annotate-tokens=%s:4:1:165:38 %s | FileCheck %s

// RUN: c-index-test -test-annotate-tokens=%s:50:1:55:1 %s | FileCheck %s -check-prefix=CHECK-RANGE1
// CHECK-RANGE1: Keyword: "void" [50:1 - 50:5] FunctionDecl=func1:50:6
// CHECK-RANGE1: Identifier: "func1" [50:6 - 50:11] FunctionDecl=func1:50:6
// CHECK-RANGE1: Punctuation: "(" [50:11 - 50:12] FunctionDecl=func1:50:6
// CHECK-RANGE1: Keyword: "void" [50:12 - 50:16] FunctionDecl=func1:50:6
// CHECK-RANGE1: Punctuation: ")" [50:16 - 50:17] FunctionDecl=func1:50:6
// CHECK-RANGE1: Punctuation: ";" [50:17 - 50:18]
// CHECK-RANGE1: Identifier: "TYPE_INST" [52:1 - 52:10] macro expansion=TYPE_INST:43:9
// CHECK-RANGE1: Punctuation: "(" [52:10 - 52:11]
// CHECK-RANGE1: Identifier: "Foo" [52:11 - 52:14] VarDecl=Foo:52:11 (Definition)
// CHECK-RANGE1: Punctuation: "," [52:14 - 52:15]
// CHECK-RANGE1: Punctuation: "." [53:5 - 53:6] UnexposedExpr=
// CHECK-RANGE1: Identifier: "x" [53:6 - 53:7] MemberRef=x:52:1
// CHECK-RANGE1: Punctuation: "=" [53:8 - 53:9] UnexposedExpr=
// CHECK-RANGE1: Literal: "0" [53:10 - 53:11] IntegerLiteral=
// CHECK-RANGE1: Punctuation: "," [53:11 - 53:12] InitListExpr=
// CHECK-RANGE1: Punctuation: "." [54:5 - 54:6] UnexposedExpr=
// CHECK-RANGE1: Identifier: "y" [54:6 - 54:7] MemberRef=y:52:1
// CHECK-RANGE1: Punctuation: "=" [54:8 - 54:9] UnexposedExpr=
// CHECK-RANGE1: Literal: "1" [54:10 - 54:11] IntegerLiteral=
// CHECK-RANGE1: Punctuation: "," [54:11 - 54:12] InitListExpr=

// RUN: c-index-test -test-annotate-tokens=%s:54:1:70:1 %s | FileCheck %s -check-prefix=CHECK-RANGE2
// CHECK-RANGE2: Punctuation: "." [54:5 - 54:6] UnexposedExpr=
// CHECK-RANGE2: Identifier: "y" [54:6 - 54:7] MemberRef=y:52:1
// CHECK-RANGE2: Punctuation: "=" [54:8 - 54:9] UnexposedExpr=
// CHECK-RANGE2: Literal: "1" [54:10 - 54:11] IntegerLiteral=
// CHECK-RANGE2: Punctuation: "," [54:11 - 54:12] InitListExpr=
// CHECK-RANGE2: Punctuation: "." [55:5 - 55:6] UnexposedExpr=
// CHECK-RANGE2: Identifier: "z" [55:6 - 55:7] MemberRef=z:52:1
// CHECK-RANGE2: Punctuation: "=" [55:8 - 55:9] UnexposedExpr=
// CHECK-RANGE2: Literal: "2" [55:10 - 55:11] IntegerLiteral=
// CHECK-RANGE2: Punctuation: "," [55:11 - 55:12] InitListExpr=
// CHECK-RANGE2: Punctuation: ")" [56:1 - 56:2]
// CHECK-RANGE2: Punctuation: ";" [56:2 - 56:3]
// CHECK-RANGE2: Keyword: "void" [58:1 - 58:5] FunctionDecl=func2:58:6
// CHECK-RANGE2: Identifier: "func2" [58:6 - 58:11] FunctionDecl=func2:58:6
// CHECK-RANGE2: Punctuation: "(" [58:11 - 58:12] FunctionDecl=func2:58:6
// CHECK-RANGE2: Keyword: "void" [58:12 - 58:16] FunctionDecl=func2:58:6
// CHECK-RANGE2: Punctuation: ")" [58:16 - 58:17] FunctionDecl=func2:58:6
// CHECK-RANGE2: Punctuation: ";" [58:17 - 58:18]

// CHECK-RANGE2: Identifier: "reg" [68:3 - 68:6] DeclRefExpr=reg:67:7
// CHECK-RANGE2: Punctuation: "." [68:6 - 68:7] MemberRefExpr=field:62:9
// CHECK-RANGE2: Identifier: "field" [68:7 - 68:12] MemberRefExpr=field:62:9
