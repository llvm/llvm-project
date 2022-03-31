// RUN: %clang_cc1 -triple=i686-apple-darwin9 -emit-llvm -o - %s | FileCheck %s

int main(void) {
  int n;
  
  const char * inc = @encode(int[]);
// CHECK: ^i
// CHECK-NOT: ^i
  const char * vla = @encode(int[n]);
// CHECK: [0i]
// CHECK-NOT: [0i]
}

// PR3648
int a[sizeof(@encode(int)) == 2 ? 1 : -1]; // Type is char[2]
const char *B = @encode(int);
char (*c)[2] = &@encode(int); // @encode is an lvalue

char d[] = @encode(int);   // infer size.
char e[1] = @encode(int);  // truncate
char f[2] = @encode(int);  // fits
char g[3] = @encode(int);  // zero fill

