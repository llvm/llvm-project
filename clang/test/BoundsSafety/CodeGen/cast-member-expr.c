

// RUN: %clang_cc1 -O0 -fbounds-safety -emit-llvm %s -o /dev/null
// RUN: %clang_cc1 -O0 -fbounds-safety -x objective-c -fbounds-attributes-objc-experimental -emit-llvm %s -o /dev/null

struct ctx { int x; };

void foo(void) {
  int *p = 0;
  ((struct ctx *)p)->x = 0;
}
