// RUN: %clang_cc1 -emit-llvm < %s

int b();
int main() { return b(b); }
int b(int (*f)()){
  return 0;
}