// RUN: %clang_cc1 -emit-llvm %s  -o /dev/null


struct S { };

int xxxx(int a) {
  struct S comps[a];
  comps[0];
}

