struct Foo {
    // Those are not explicit constructors
    Foo(int);
    explicit(false) Foo(float);

    // Those are explicit constructors
    explicit Foo(double);
    explicit(true) Foo(char);
};

// RUN: c-index-test -test-print-type --std=c++20 %s | FileCheck %s
// CHECK: StructDecl=Foo:1:8 (Definition) [type=Foo] [typekind=Record] [isPOD=0]
// CHECK: CXXConstructor=Foo:3:5 (converting constructor) [type=void (int)] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [int] [Int]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXConstructor=Foo:4:21 (converting constructor) [type=void (float)] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [float] [Float]] [isPOD=0] [isAnonRecDecl=0]
// CXXConstructor=Foo:7:20 (explicit) [type=void (double)] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [double] [Double]] [isPOD=0] [isAnonRecDecl=0]
// CXXConstructor=Foo:8:20 (explicit) [type=void (char)] [typekind=FunctionProto] [resulttype=void] [resulttypekind=Void] [args= [char] [Char_S]] [isPOD=0] [isAnonRecDecl=0]
