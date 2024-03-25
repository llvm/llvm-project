// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o %t %s
// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -emit-llvm -o %t %s

@interface Super {
  int ivar_super_a : 5;
}
@end

@interface A : Super {
@public
  int ivar_a : 5;
}
@end

int f0(A *a) {
  return a->ivar_a;
}

@interface A () {
@public
  int ivar_ext_a : 5;
  int ivar_ext_b : 5;
}@end

int f1(A *a) {
  return a->ivar_ext_a + a->ivar_a;
}

@interface A () {
@public
  int ivar_ext2_a : 5;
  int ivar_ext2_b : 5;
}@end

int f2(A* a) {
  return a->ivar_ext2_a + a->ivar_ext_a + a->ivar_a;
}

@implementation A {
@public
  int ivar_b : 5;
  int ivar_c : 5;
  int ivar_d : 5;
}
@end

int f3(A *a) {  
  return a->ivar_d + a->ivar_ext2_a + a->ivar_ext_a + a->ivar_a;
}

