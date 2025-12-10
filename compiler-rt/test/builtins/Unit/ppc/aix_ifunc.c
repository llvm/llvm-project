// RUN: %clang %s -fno-integrated-as -Wl,-bmap:%t.map
// RUN: FileCheck %s < %t.map
// RUN: %clang %s -m64 -fno-integrated-as -Wl,-bmap:%t.map
// RUN: FileCheck %s --check-prefix=CHECK64 < %t.map

// CHECK: [[#%X,ADDR:]]          RW LD S{{.*}}   {__start___ifunc_sec}
// CHECK: [[#ADDR+8]]          RW LD S{{.*}}   {__stop___ifunc_sec}
// CHECK64: [[#%X,ADDR:]]          RW LD S{{.*}}   {__start___ifunc_sec}
// CHECK64: [[#ADDR+16]]          RW LD S{{.*}}   {__stop___ifunc_sec}


static int my_foo() { return 5; }
static void* foo_resolver() { return &my_foo; };

__attribute__ ((ifunc ("foo_resolver")))
int foo();
int main() { return foo(); }
