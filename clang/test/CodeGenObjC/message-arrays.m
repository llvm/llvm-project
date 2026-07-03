// RUN: %clang_cc1 -emit-llvm -o %t %s

void f0(id a) {
  // This should have an implicit cast
  [ a print: "hello" ];
}

@interface A
-(void) m: (int) arg0, ...;
@end

void f1(A *a) {
  // This should also get an implicit cast (for the vararg)
  [a m: 1, "test"];
}
