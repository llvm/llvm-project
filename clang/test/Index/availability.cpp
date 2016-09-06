void foo() = delete;

struct Foo {
  int foo() = delete;
  Foo() = delete;
};


// RUN: c-index-test -test-print-type --std=c++11 %s | FileCheck %s
// CHECK: FunctionDecl=foo:1:6 (unavailable) [type=void ()] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: StructDecl=Foo:3:8 (Definition) [type=Foo] [typekind=Record] [isPOD=1]
// CHECK: CXXMethod=foo:4:7 (unavailable) [type=int (){{.*}}] [typekind=FunctionProto] [resulttype=int] [resulttypekind=Int] [isPOD=0]
// CHECK: CXXConstructor=Foo:5:3 (unavailable) (default constructor) [type=void (){{.*}}] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
