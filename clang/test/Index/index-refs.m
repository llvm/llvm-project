
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

// RUN: c-index-test -index-file %s | FileCheck %s
// CHECK: [indexEntityReference]: kind: objc-protocol | name: Prot | {{.*}} | loc: 12:27
// CHECK: [indexEntityReference]: kind: struct | name: FooS | {{.*}} | loc: 13:18
