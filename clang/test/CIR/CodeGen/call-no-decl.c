// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=gnu89 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

// Regression test for https://github.com/llvm/llvm-project/issues/182175
// A function called with no prior declaration must get a variadic CIR function
// type so that multiple call sites with different argument counts do not
// trigger a verifier error.

char temp[] = "some str";
int foo(const char *);

int bar(void) {
  int t = foo(temp);
  printf("Hello %d!\n", t);
  printf("Works!\n");
  return 0;
}

// CHECK: cir.func private @printf(!cir.ptr<!s8i>, ...) -> !s32i
// CHECK: cir.call @printf({{.*}}, {{.*}}) : (!cir.ptr<!s8i>, !s32i) -> !s32i
// CHECK: cir.call @printf({{.*}}) : (!cir.ptr<!s8i>) -> !s32i