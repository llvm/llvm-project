// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: not %clang %s -c -emit-llvm -o %t.dir
// RUN: test -d %t.dir

int main() { return 0; }
