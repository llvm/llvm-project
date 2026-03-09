struct Foo {
  void normal();
  void c() const;
  void v() volatile;
  void cv() const volatile;
  void r() __restrict;
  void cvr() const volatile __restrict;
  constexpr int baz() { return 1; }
};
constexpr int x = 42;
int y = 1;

// RUN: c-index-test -test-print-type --std=c++14 %s | FileCheck %s
// CHECK: CXXMethod=normal:2:8 [type=void ()] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXMethod=c:3:8 (const) [type=void () const] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXMethod=v:4:8 (volatile) [type=void () volatile] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXMethod=cv:5:8 (const) (volatile) [type=void () const volatile] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXMethod=r:6:8 (restrict) [type=void () __restrict] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXMethod=cvr:7:8 (const) (volatile) (restrict) [type=void () const volatile __restrict] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXMethod=baz:8:17 (Definition) (constexpr) [type=int ()] [typekind=FunctionProto] [resulttype=int] [resulttypekind=Int] [isPOD=0]
// CHECK: VarDecl=x:10:15 (Definition) (constexpr) [type=const int] [typekind=Int] const [isPOD=1]
// CHECK: VarDecl=y:11:5 (Definition) [type=int] [typekind=Int] [isPOD=1]
