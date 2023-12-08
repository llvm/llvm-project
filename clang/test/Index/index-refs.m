
@class Protocol;

@protocol Prot
@end

struct FooS {
  int x;
};

void foo() {
  Protocol *p = @protocol(Prot);
  @encode(struct FooS);
}

@interface I
+(void)clsMeth;
@end

void foo2() {
  [I clsMeth];
}

@protocol ForwardProt;

// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK: [indexEntityReference]: kind: objc-protocol | name: Prot | {{.*}} | loc: 12:27
// CHECK: [indexEntityReference]: kind: struct | name: FooS | {{.*}} | loc: 13:18
// CHECK: [indexEntityReference]: kind: objc-class | name: I | {{.*}} | loc: 21:4

// CHECK: [indexDeclaration]: kind: objc-protocol | name: ForwardProt | {{.*}} | loc: 24:11
// CHECK-NEXT: <ObjCContainerInfo>: kind: forward-ref
