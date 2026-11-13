// RUN: env SDKROOT="/" %clang -Xclang -source-date-epoch -Xclang 1613676318 -emit-llvm -S -o - -x c - < %s | grep -v DIFile > %t1.ll
// RUN: env SDKROOT="/" %clang -emit-ast -o %t.ast %s
// RUN: env SDKROOT="/" %clang -Xclang -source-date-epoch -Xclang 1613676318 -emit-llvm -S -o - -x ast - < %t.ast | grep -v DIFile > %t2.ll
// RUN: diff %t1.ll %t2.ll

int main(void) {
  return 0;
}
