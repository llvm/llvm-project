// RUN: %clang_cc1 -emit-llvm -triple=i686-apple-darwin8 -o %t %s
// RUNX: %clang_cc1 -emit-llvm -o %t %s

@interface A
 -(void) setOk:(int)arg;
 -(int) ok;

 -(void) setX:(int)arg;
 -(int) x;
@end

void f0(A *a) {
   a.x = 1;   
   a.ok = a.x;
}

