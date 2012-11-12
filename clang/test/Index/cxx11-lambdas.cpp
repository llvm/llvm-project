// Test is line- and column-sensitive; see below.

typedef int Integer;
struct X {
  void f() {
    int localA, localB;
    auto lambda = [&localA, localB] (Integer x) -> Integer {
      return localA + localB + x;
    };
  }
};

// RUN: c-index-test -test-load-source all -std=c++11 %s | FileCheck -check-prefix=CHECK-LOAD %s
// CHECK-LOAD: cxx11-lambdas.cpp:7:19: LambdaExpr= Extent=[7:19 - 9:6]
// CHECK-LOAD: cxx11-lambdas.cpp:7:21: VariableRef=localA:6:9 Extent=[7:21 - 7:27]
// CHECK-LOAD: cxx11-lambdas.cpp:7:29: VariableRef=localB:6:17 Extent=[7:29 - 7:35]
// CHECK-LOAD: cxx11-lambdas.cpp:7:52: TypeRef=Integer:3:13 Extent=[7:52 - 7:59]
// CHECK-LOAD: cxx11-lambdas.cpp:7:46: ParmDecl=x:7:46 (Definition) Extent=[7:38 - 7:47]
// CHECK-LOAD: cxx11-lambdas.cpp:7:38: TypeRef=Integer:3:13 Extent=[7:38 - 7:45]
// CHECK-LOAD: cxx11-lambdas.cpp:7:60: CompoundStmt= Extent=[7:60 - 9:6]
// CHECK-LOAD: cxx11-lambdas.cpp:8:7: ReturnStmt= Extent=[8:7 - 8:33]
// CHECK-LOAD: cxx11-lambdas.cpp:8:14: DeclRefExpr=localA:6:9 Extent=[8:14 - 8:20]
// CHECK-LOAD: cxx11-lambdas.cpp:8:23: DeclRefExpr=localB:6:17 Extent=[8:23 - 8:29]
// CHECK-LOAD: cxx11-lambdas.cpp:8:32: DeclRefExpr=x:7:46 Extent=[8:32 - 8:33]

// RUN: env CINDEXTEST_INDEXLOCALSYMBOLS=1 c-index-test -index-file -std=c++11 %s | FileCheck -check-prefix=CHECK-INDEX %s
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: localA | USR: c:cxx11-lambdas.cpp@100@S@X@F@f#@localA | lang: C | cursor: VariableRef=localA:6:9 | loc: 7:21
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: localB | USR: c:cxx11-lambdas.cpp@100@S@X@F@f#@localB | lang: C | cursor: VariableRef=localB:6:17 | loc: 7:29
// CHECK-INDEX: [indexEntityReference]: kind: typedef | name: Integer | USR: c:cxx11-lambdas.cpp@51@T@Integer | lang: C | cursor: TypeRef=Integer:3:13 | loc: 7:52
// CHECK-INDEX: [indexEntityReference]: kind: typedef | name: Integer | USR: c:cxx11-lambdas.cpp@51@T@Integer | lang: C | cursor: TypeRef=Integer:3:13 | loc: 7:38
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: localA | USR: c:cxx11-lambdas.cpp@100@S@X@F@f#@localA | lang: C | cursor: DeclRefExpr=localA:6:9 | loc: 8:14
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: localB | USR: c:cxx11-lambdas.cpp@100@S@X@F@f#@localB | lang: C | cursor: DeclRefExpr=localB:6:17 | loc: 8:23
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: x | USR: c:cxx11-lambdas.cpp@157@S@X@F@f#@Ca@F@operator()#I#1@x | lang: C | cursor: DeclRefExpr=x:7:46 | loc: 8:32
