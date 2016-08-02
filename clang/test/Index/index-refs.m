
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

// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK: [indexEntityReference]: kind: objc-protocol | name: Prot | {{.*}} | loc: 12:27
// CHECK: [indexEntityReference]: kind: struct | name: FooS | {{.*}} | loc: 13:18
// CHECK: [indexEntityReference]: kind: objc-class | name: I | {{.*}} | loc: 21:4
