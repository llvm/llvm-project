// RUN: %clangxx_hwasan -mllvm -hwasan-use-stack-safety=0 %s -o %t
// RUN: %run %t
//
// REQUIRES: pointer-tagging

__attribute__((noinline)) int bar(int X) { return X; }

__attribute__((noinline)) int foo(int X) {
  volatile int A = 5;
  [[clang::musttail]] return bar(X + A);
}

int main(int Argc, char *Argv[]) { return foo(Argc) != 6; }
