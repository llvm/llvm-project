
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG


#include <x86intrin.h>

void test_m_prefetch_w(void *p) {
  // CIR-LABEL: test_m_prefetch_w
  // LLVM-LABEL: test_m_prefetch_w
  // OGCG-LABEL: test_m_prefetch_w
  return _m_prefetchw(p);
  // CIR: cir.prefetch write locality(3) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 1, i32 3, i32 1)
}

void test_m_prefetch(void *p) {
  // CIR-LABEL: test_m_prefetch
  // LLVM-LABEL: test_m_prefetch
  // OGCG-LABEL: test_m_prefetch
  return _m_prefetch(p);
  // CIR: cir.prefetch read locality(3) %{{.*}} : !cir.ptr<!void>
  // LLVM: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
  // OGCG: call void @llvm.prefetch.p0(ptr {{.*}}, i32 0, i32 3, i32 1)
}
