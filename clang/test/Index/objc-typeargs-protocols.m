
@interface TestA
@end

@interface TestB
@end

@protocol Bar
@end

@interface Base
@end

@interface Foo<FirstType, SecondType> : Base
@end

Foo *a;
Foo<TestA *, TestB *> *b;
Foo<Bar> *c;
Foo<TestA *, TestB *><Bar> *d;
id<Bar> e;

// RUN: c-index-test -test-print-type %s | FileCheck %s
// CHECK: VarDecl=a:17:6 [type=Foo *] [typekind=ObjCObjectPointer] [basetype=Foo] [basekind=ObjCInterface] [isPOD=1] [pointeetype=Foo] [pointeekind=ObjCInterface]
// CHECK: VarDecl=b:18:24 [type=Foo<TestA *,TestB *> *] [typekind=ObjCObjectPointer] [basetype=Foo] [basekind=ObjCInterface] [typeargs= [TestA *] [ObjCObjectPointer] [TestB *] [ObjCObjectPointer]] [isPOD=1] [pointeetype=Foo<TestA *,TestB *>] [pointeekind=ObjCObject]
// CHECK: VarDecl=c:19:11 [type=Foo<Bar> *] [typekind=ObjCObjectPointer] [basetype=Foo] [basekind=ObjCInterface] [protocols=ObjCProtocolDecl=Bar:8:11 (Definition)] [isPOD=1] [pointeetype=Foo<Bar>] [pointeekind=ObjCObject]
// CHECK: VarDecl=d:20:29 [type=Foo<TestA *,TestB *><Bar> *] [typekind=ObjCObjectPointer] [basetype=Foo] [basekind=ObjCInterface] [typeargs= [TestA *] [ObjCObjectPointer] [TestB *] [ObjCObjectPointer]] [protocols=ObjCProtocolDecl=Bar:8:11 (Definition)] [isPOD=1] [pointeetype=Foo<TestA *,TestB *><Bar>] [pointeekind=ObjCObject]
// CHECK: VarDecl=e:21:9 [type=id<Bar>] [typekind=ObjCObjectPointer] [basetype=id] [basekind=ObjCId] [protocols=ObjCProtocolDecl=Bar:8:11 (Definition)] [isPOD=1] [pointeetype=id<Bar>] [pointeekind=ObjCObject]
