// Test is line- and column-sensitive; see below.

typedef int Integer;
struct X {
  void f() {
    int localA, localB;
    auto lambda = [ptr = &localA, copy = localB] (Integer x) -> Integer {
      return *ptr + copy + x;
    };
  }
};

// RUN: c-index-test -test-load-source all -std=c++14 %s | FileCheck -check-prefix=CHECK-LOAD %s
// CHECK-LOAD: cxx14-lambdas.cpp:7:5: DeclStmt= Extent=[7:5 - 9:7]
// CHECK-LOAD: cxx14-lambdas.cpp:7:10: VarDecl=lambda:7:10 (Definition) Extent=[7:5 - 9:6]
// CHECK-LOAD: cxx14-lambdas.cpp:7:19: UnexposedExpr= Extent=[7:19 - 9:6]
// CHECK-LOAD: cxx14-lambdas.cpp:7:19: CallExpr= Extent=[7:19 - 9:6]
// CHECK-LOAD: cxx14-lambdas.cpp:7:19: UnexposedExpr= Extent=[7:19 - 9:6]
// CHECK-LOAD: cxx14-lambdas.cpp:7:19: LambdaExpr= Extent=[7:19 - 9:6]
// CHECK-LOAD: cxx14-lambdas.cpp:7:20: VariableRef=ptr:7:20 Extent=[7:20 - 7:23]
// CHECK-LOAD: cxx14-lambdas.cpp:7:35: VariableRef=copy:7:35 Extent=[7:35 - 7:39]
// CHECK-LOAD: cxx14-lambdas.cpp:7:27: DeclRefExpr=localA:6:9 Extent=[7:27 - 7:33]
// CHECK-LOAD: cxx14-lambdas.cpp:7:42: DeclRefExpr=localB:6:17 Extent=[7:42 - 7:48]
// CHECK-LOAD: cxx14-lambdas.cpp:7:59: ParmDecl=x:7:59 (Definition) Extent=[7:51 - 7:60]
// CHECK-LOAD: cxx14-lambdas.cpp:7:51: TypeRef=Integer:3:13 Extent=[7:51 - 7:58]
// CHECK-LOAD: cxx14-lambdas.cpp:7:65: TypeRef=Integer:3:13 Extent=[7:65 - 7:72]
// CHECK-LOAD: cxx14-lambdas.cpp:7:73: CompoundStmt= Extent=[7:73 - 9:6]
// CHECK-LOAD: cxx14-lambdas.cpp:8:7: ReturnStmt= Extent=[8:7 - 8:29]

// RUN: env CINDEXTEST_INDEXLOCALSYMBOLS=1 c-index-test -index-file -std=c++14 %s | FileCheck -check-prefix=CHECK-INDEX %s
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: ptr | USR: c:cxx14-lambdas.cpp@139@S@X@F@f#@Sa@F@operator()#I#1@ptr | lang: C | cursor: VariableRef=ptr:7:20 | loc: 7:20
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: copy | USR: c:cxx14-lambdas.cpp@154@S@X@F@f#@Sa@F@operator()#I#1@copy | lang: C | cursor: VariableRef=copy:7:35 | loc: 7:35
// CHECK-INDEX: [indexDeclaration]: kind: variable | name: x | USR: c:cxx14-lambdas.cpp@170@S@X@F@f#@Sa@F@operator()#I#1@x | lang: C | cursor: ParmDecl=x:7:59 (Definition) | loc: 7:59
// CHECK-INDEX: [indexEntityReference]: kind: typedef | name: Integer | USR: c:cxx14-lambdas.cpp@T@Integer | lang: C | cursor: TypeRef=Integer:3:13 | loc: 7:51
// CHECK-INDEX: [indexEntityReference]: kind: typedef | name: Integer | USR: c:cxx14-lambdas.cpp@T@Integer | lang: C | cursor: TypeRef=Integer:3:13 | loc: 7:65
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: ptr | USR: c:cxx14-lambdas.cpp@139@S@X@F@f#@Sa@F@operator()#I#1@ptr | lang: C | cursor: DeclRefExpr=ptr:7:20 | loc: 8:15
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: copy | USR: c:cxx14-lambdas.cpp@154@S@X@F@f#@Sa@F@operator()#I#1@copy | lang: C | cursor: DeclRefExpr=copy:7:35 | loc: 8:21
// CHECK-INDEX: [indexEntityReference]: kind: variable | name: x | USR: c:cxx14-lambdas.cpp@170@S@X@F@f#@Sa@F@operator()#I#1@x | lang: C | cursor: DeclRefExpr=x:7:59 | loc: 8:28
