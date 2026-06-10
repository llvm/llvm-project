// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - -Wno-deprecated-attributes | FileCheck %s --check-prefix=SPIR
// RUN: %clang_cc1 -triple x86_64 -fsycl-is-device -disable-llvm-passes -emit-llvm %s -o - -Wno-deprecated-attributes | FileCheck %s --check-prefix=X86

// REQUIRES: x86-registered-target

void foo(int [[clang::sycl_global]] *);
void foo(int [[clang::sycl_local]] *);
void foo(int [[clang::sycl_private]] *);
void foo(int [[clang::sycl_generic]] *);
void foo(int [[clang::sycl_constant]] *);
void foo(int *);

// SPIR: declare spir_func void @_Z3fooPU3AS1i(ptr addrspace(1) noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS3i(ptr addrspace(3) noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS0i(ptr noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS4i(ptr addrspace(4) noundef) #1
// SPIR: declare spir_func void @_Z3fooPU3AS2i(ptr addrspace(2) noundef) #1
// SPIR: declare spir_func void @_Z3fooPi(ptr addrspace(4) noundef) #1

// X86: declare void @_Z3fooPU8SYglobali(ptr noundef) #1
// X86: declare void @_Z3fooPU7SYlocali(ptr noundef) #1
// X86: declare void @_Z3fooPU9SYprivatei(ptr noundef) #1
// X86: declare void @_Z3fooPU9SYgenerici(ptr noundef) #1
// X86: declare void @_Z3fooPU10SYconstanti(ptr noundef) #1
// X86: declare void @_Z3fooPi(ptr noundef) #1

[[clang::sycl_external]] void test() {
  int [[clang::sycl_global]] *glob;
  int [[clang::sycl_local]] *loc;
  int [[clang::sycl_private]] *priv;
  int [[clang::sycl_generic]] *gen;
  int [[clang::sycl_constant]] *cnst;
  int *def;
  foo(glob);
  foo(loc);
  foo(priv);
  foo(gen);
  foo(cnst);
  foo(def);
}
