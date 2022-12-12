// UNSUPPORTED: -zos, target={{.*}}-aix{{.*}}
// RUN: clang-import-test -x objective-c++ -import %S/Inputs/S.m -expression %s
@class D;

@interface B {
  int x;
  int y;
}
@end

@interface D : B {
  int z;
}
-(int)n;
@end

void expr() {
  C *c;
  int i = [c m];
  D *d;
  int j = [d n] + d->x;
}
