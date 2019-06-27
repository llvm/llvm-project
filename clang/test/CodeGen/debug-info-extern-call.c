// RUN: %clang -Xclang -femit-debug-entry-values -g -O2 -target x86_64-none-linux-gnu -S -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-EXT
// CHECK-EXT: !DISubprogram(name: "fn1"

// RUN: %clang -g -O2 -target x86_64-none-linux-gnu -S -emit-llvm %s -o - | FileCheck %s
// CHECK-NOT: !DISubprogram(name: "fn1"

extern int fn1(int a, int b);

int fn2 () {
  int x = 4, y = 5;
  int res = fn1(x, y);

  return res;
}

