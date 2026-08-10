struct Foo {
    // Those are not explicit conversion functions
    operator int();
    explicit(false) operator float();

    // Those are explicit conversion functions
    explicit operator double();
    explicit(true) operator unsigned char();
};

// RUN: c-index-test -test-print-type --std=c++20 %s | FileCheck %s
// CHECK: StructDecl=Foo:1:8 (Definition) [type=Foo] [typekind=Record]
// CHECK: CXXConversion=operator int:3:5 [type=int ()] [typekind=FunctionProto] [resulttype=int] [resulttypekind=Int] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXConversion=operator float:4:21 [type=float ()] [typekind=FunctionProto] [resulttype=float] [resulttypekind=Float] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXConversion=operator double:7:14 (explicit) [type=double ()] [typekind=FunctionProto] [resulttype=double] [resulttypekind=Double] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXConversion=operator unsigned char:8:20 (explicit) [type=unsigned char ()] [typekind=FunctionProto] [resulttype=unsigned char] [resulttypekind=UChar] [isPOD=0] [isAnonRecDecl=0]
