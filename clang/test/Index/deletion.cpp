struct Foo {
  int foo() = delete;
  int bar();
  Foo() = delete;
  Foo(int);
};

int foo(int);
int foo(double) = delete;

template <typename T>
void processPointer(T* ptr);

template <>
void processPointer<void>(void* ptr) = delete;

// RUN: c-index-test -test-print-type --std=c++11 %s | FileCheck %s
// CHECK: StructDecl=Foo:1:8 (Definition) [type=Foo] [typekind=Record] [isPOD=1]
// CHECK: CXXMethod=foo:2:7 (unavailable) (deleted) [type=int (){{.*}}] [typekind=FunctionProto] [resulttype=int] [resulttypekind=Int] [isPOD=0]
// CHECK: CXXMethod=bar:3:7 [type=int (){{.*}}] [typekind=FunctionProto] [resulttype=int] [resulttypekind=Int] [isPOD=0]
// CHECK: CXXConstructor=Foo:4:3 (unavailable) (default constructor) (deleted) [type=void (){{.*}}] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0]
// CHECK: CXXConstructor=Foo:5:3 (converting constructor) [type=void (int){{.*}}] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [int] [Int]] [isPOD=0]

// CHECK: FunctionDecl=foo:8:5 [type=int (int)] [typekind=FunctionProto] [resulttype=int] [resulttypekind=Int] [args= [int] [Int]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: ParmDecl=:8:12 (Definition) [type=int] [typekind=Int] [isPOD=1] [isAnonRecDecl=0]
// CHECK: FunctionDecl=foo:9:5 (unavailable) (deleted) [type=int (double)] [typekind=FunctionProto] [resulttype=int] [resulttypekind=Int] [args= [double] [Double]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: ParmDecl=:9:15 (Definition) [type=double] [typekind=Double] [isPOD=1] [isAnonRecDecl=0]

// CHECK: FunctionTemplate=processPointer:12:6 [type=void (T *)] [typekind=FunctionProto] [canonicaltype=void (type-parameter-0-0 *)] [canonicaltypekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [isPOD=0] [isAnonRecDecl=0]
// CHECK: FunctionDecl=processPointer:15:6 (unavailable) (deleted) [Specialization of processPointer:12:6] [Template arg 0: kind: 1, type: void] [type=void (void *)] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [void *] [Pointer]] [isPOD=0] [isAnonRecDecl=0]
