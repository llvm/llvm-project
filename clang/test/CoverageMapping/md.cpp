// RUN: %clang_cc1 -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm-only -std=c++11 %s | FileCheck %s

#define BREAK break

enum class MD {
  Val1,
  Val2
};

void nop() {}

// CHECK: foo
// CHECK-NEXT: File 0, [[@LINE+1]]:16 -> {{[0-9]+}}:2 = #0
void foo(MD i) {
  switch (i) {
  #define HANDLE_MD(X)                                          \
  case MD::X:                                                   \
    break;
  #include "Inputs/md.def"
  default:
    BREAK;
  }

  if (false)
    nop();
  #define HANDLE_MD(X) else if (i == MD::X) { nop(); }
  #include "Inputs/md.def"
}

int main(int argc, const char *argv[]) {
  foo(MD::Val1);
  return 0;
}
