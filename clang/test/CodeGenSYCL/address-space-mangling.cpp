// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=SPIR
// RUN: %clang_cc1 -triple x86_64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s --check-prefix=X86

// REQUIRES: x86-registered-target

void foo(__attribute__((opencl_global)) int *);
void foo(__attribute__((opencl_local)) int *);
void foo(__attribute__((opencl_private)) int *);
void foo(int *);

// SPIR: declare spir_func void @_Z3fooPU3AS1i(i32 addrspace(1)*) #1
// SPIR: declare spir_func void @_Z3fooPU3AS3i(i32 addrspace(3)*) #1
// SPIR: declare spir_func void @_Z3fooPU3AS0i(i32*) #1
// SPIR: declare spir_func void @_Z3fooPi(i32 addrspace(4)*) #1

// X86: declare void @_Z3fooPU8SYglobali(i32*) #1
// X86: declare void @_Z3fooPU7SYlocali(i32*) #1
// X86: declare void @_Z3fooPU9SYprivatei(i32*) #1
// X86: declare void @_Z3fooPi(i32*) #1

void test() {
  __attribute__((opencl_global)) int *glob;
  __attribute__((opencl_local)) int *loc;
  __attribute__((opencl_private)) int *priv;
  int *def;
  foo(glob);
  foo(loc);
  foo(priv);
  foo(def);
}
