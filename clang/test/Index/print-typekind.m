@interface Foo
@property (readonly) id x;
-(int) mymethod;
-(const id) mymethod2:(id)x blah:(Class)y boo:(SEL)z;
@end

// RUN: c-index-test -test-print-typekind %s | FileCheck %s
// CHECK: ObjCPropertyDecl=x:2:25 typekind=ObjCId [canonical=ObjCObjectPointer]
// CHECK: ObjCInstanceMethodDecl=mymethod:3:8 typekind=Invalid [result=Int]
// CHECK: ObjCInstanceMethodDecl=mymethod2:blah:boo::4:13 typekind=Invalid [result=ObjCId] [args= ObjCId ObjCClass ObjCSel]
