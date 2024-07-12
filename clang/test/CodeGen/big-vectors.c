// RUN: %clang_cc1 -O0 -triple x86_64 %s -emit-llvm -o - | FileCheck --check-prefixes=x86 %s
// RUN: %clang_cc1 -O0 -triple spir64 %s -emit-llvm -o - | FileCheck --check-prefixes=SPIR %s

typedef float fvec __attribute__((ext_vector_type(5120)));
fvec foo(fvec a) {
  fvec c;
  return c;
}
// x86-DAG: define{{.*}}@foo{{.*}}sret(<5120 x float>) align 16384{{.*}}byval(<5120 x float>) align 16384
// SPIR: define{{.*}}@foo({{.*}}sret(<5120 x float>) align 16384{{.*}}byval(<5120 x float>) align 16384

void bar() {
  fvec a;
  fvec c = foo(a);
// x86-DAG: call void @foo({{.*}}sret(<5120 x float>) align 16384{{.*}}byval(<5120 x float>) align 16384
// SPIR: call spir_func void @foo({{.*}}sret(<5120 x float>) align 16384{{.*}}byval(<5120 x float>) align 16384
}
