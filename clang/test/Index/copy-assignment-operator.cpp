struct Foo {
    // Those are copy-assignment operators
    bool operator=(const Foo&);
    bool operator=(Foo&);
    bool operator=(volatile Foo&);
    bool operator=(const volatile Foo&);
    bool operator=(Foo);

    // Those are not copy-assignment operators
    template<typename T>
    bool operator=(const T&);
    bool operator=(const bool&);
    bool operator=(char&);
    bool operator=(volatile unsigned int&);
    bool operator=(const volatile unsigned char&);
    bool operator=(int);
};

// Positive-check that the recognition works for templated classes too
template <typename T>
class Bar {
    bool operator=(const Bar&);
    bool operator=(Bar<T>&);
    bool operator=(volatile Bar&);
    bool operator=(const volatile Bar<T>&);
    bool operator=(Bar);
};

// RUN: c-index-test -test-print-type --std=c++11 %s | FileCheck %s
// CHECK: StructDecl=Foo:1:8 (Definition) [type=Foo] [typekind=Record] [isPOD=0]
// CHECK: CXXMethod=operator=:3:10 (copy-assignment operator) [type=bool (const Foo &)] [typekind=FunctionProto] [canonicaltype=bool (const Foo &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [const Foo &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:4:10 (copy-assignment operator) [type=bool (Foo &)] [typekind=FunctionProto] [canonicaltype=bool (Foo &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [Foo &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:5:10 (copy-assignment operator) [type=bool (volatile Foo &)] [typekind=FunctionProto] [canonicaltype=bool (volatile Foo &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [volatile Foo &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:6:10 (copy-assignment operator) [type=bool (const volatile Foo &)] [typekind=FunctionProto] [canonicaltype=bool (const volatile Foo &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [const volatile Foo &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:7:10 (copy-assignment operator) [type=bool (Foo)] [typekind=FunctionProto] [canonicaltype=bool (Foo)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [Foo] [Elaborated]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: FunctionTemplate=operator=:11:10 [type=bool (const T &)] [typekind=FunctionProto] [canonicaltype=bool (const type-parameter-0-0 &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:12:10 [type=bool (const bool &)] [typekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [const bool &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:13:10 [type=bool (char &)] [typekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [char &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:14:10 [type=bool (volatile unsigned int &)] [typekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [volatile unsigned int &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:15:10 [type=bool (const volatile unsigned char &)] [typekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [const volatile unsigned char &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:16:10 [type=bool (int)] [typekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [int] [Int]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: ClassTemplate=Bar:21:7 (Definition) [type=] [typekind=Invalid] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:22:10 (copy-assignment operator) [type=bool (const Bar<T> &)] [typekind=FunctionProto] [canonicaltype=bool (const Bar<T> &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [const Bar<T> &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:23:10 (copy-assignment operator) [type=bool (Bar<T> &)] [typekind=FunctionProto] [canonicaltype=bool (Bar<T> &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [Bar<T> &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:24:10 (copy-assignment operator) [type=bool (volatile Bar<T> &)] [typekind=FunctionProto] [canonicaltype=bool (volatile Bar<T> &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [volatile Bar<T> &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:25:10 (copy-assignment operator) [type=bool (const volatile Bar<T> &)] [typekind=FunctionProto] [canonicaltype=bool (const volatile Bar<T> &)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [const volatile Bar<T> &] [LValueReference]] [isPOD=0] [isAnonRecDecl=0]
// CHECK: CXXMethod=operator=:26:10 (copy-assignment operator) [type=bool (Bar<T>)] [typekind=FunctionProto] [canonicaltype=bool (Bar<T>)] [canonicaltypekind=FunctionProto] [resulttype=bool] [resulttypekind=Bool] [args= [Bar<T>] [Elaborated]] [isPOD=0] [isAnonRecDecl=0]
