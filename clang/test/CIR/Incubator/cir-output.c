// RUN: cp %s %t.c
// RUN: %clang -fclangir -Werror -fcir-output=%t.explicit.cir -c %t.c
// RUN: FileCheck %s --input-file=%t.explicit.cir --check-prefix=CIR
// RUN: rm -f %t.cir
// RUN: %clang -fclangir -Werror -fcir-output %t.c -c -o %t.obj
// RUN: FileCheck %s --input-file=%t.cir --check-prefix=CIR

// NYI on invalid C++ ABI kind
// XFAIL: host={{.*}}-windows-msvc

struct S {
  int x;
};

int foo(void) {
  struct S s = {42};
  return s.x;
}

// CIR: module
// CIR: cir.func{{.*}}@foo
