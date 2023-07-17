// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: grep -e "^de.*objc_msgSend[0-9]*(" %t | count 1
// RUN: %clang_cc1 -DWITHDEF -emit-llvm -o %t %s
// RUN: grep -e "^de.*objc_msgSend[0-9]*(" %t | count 1

id objc_msgSend(int x);

@interface A @end

@implementation A
-(void) f0 {
  objc_msgSend(12);
}

-(void) hello {
}
@end

void f0(id x) {
  [x hello];
}

#ifdef WITHDEF
// This isn't a very good send function.
id objc_msgSend(int x) {
  return 0;
}

// rdar://6800430
void objc_assign_weak(id value, id *location) {
}

#endif
