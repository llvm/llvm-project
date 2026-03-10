// We created a constructor function in compiler-rt/lib/builtins/ppc/init_ifuncs.c
// that reads an array contained in a certain named section of the object file.
// The compiler generates extra globals (one per ifunc) in that section.
// This test is to make sure the section name in the builtins library and the
// compiler match by checking the distance between the start and end of the
// section is "2*sizeof(void*)" because there's one ifunc in the entire program.
//
// REQUIRES: target={{.*aix.*}}
// RUN: %clang_builtins %s %librt -fno-integrated-as -o a.out
// RUN: llvm-nm -Xany --numeric-sort a.out | \
// RUN:   FileCheck %s %if target-is-powerpc64 %{ --check-prefix=CHECK64 %} \
// RUN:                %else %{ --check-prefix=CHECK %}

// CHECK: [[#%x,ADDR:]] W __start___ifunc_sec
// CHECK: [[#ADDR+8]] W __stop___ifunc_sec
// CHECK64: [[#%x,ADDR:]] W __start___ifunc_sec
// CHECK64: [[#ADDR+16]] W __stop___ifunc_sec

static int my_foo() { return 5; }
static void *foo_resolver() { return &my_foo; };

__attribute__((ifunc("foo_resolver"))) int foo();
int main() { return foo(); }
