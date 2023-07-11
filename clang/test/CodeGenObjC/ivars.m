// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm -o - %s
// RUN: %clang_cc1 -fobjc-gc -emit-llvm -o - %s

@interface ITF {
@public
  unsigned field :1 ;
  _Bool boolfield :1 ;
}
@end

void foo(ITF *P) {
  P->boolfield = 1;
}

@interface R {
  struct {
    union {
      int x;
      char c;
    };
  } _union;
}
@end

@implementation R
@end
