// RUN: env SDKROOT="/" %clang -emit-llvm -S -o %t1.ll -x c - < %s
// RUN: env SDKROOT="/" %clang -emit-ast -o %t.ast %s
// RUN: env SDKROOT="/" %clang -emit-llvm -S -o %t2.ll -x ast - < %t.ast
// RUN: diff %t1.ll %t2.ll

int main() {
  return 0;
}
