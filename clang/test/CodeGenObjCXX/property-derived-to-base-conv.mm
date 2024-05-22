// RUN: %clang_cc1 -fobjc-gc -triple x86_64-apple-darwin10 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s

struct A {
  int member;
  void foo();
  A *operator->();
};
struct B : A { };

@interface BInt {
@private
  B *b;
}
- (B)value;
- (void)setValue : (B) arg;
@property B value;
@end

void g(BInt *bint) {
  bint.value.foo();
  bint.value->member = 17;
  int x = bint.value.member;
}

