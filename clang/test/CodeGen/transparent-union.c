// RUN: %clang_cc1 -Werror -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -Werror -triple i386-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -Werror -triple armv7-linux -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM
// RUN: %clang_cc1 -Werror -triple powerpc64le-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -Werror -triple aarch64-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DINFRONT -Werror -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DINFRONT -Werror -triple i386-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DINFRONT -Werror -triple armv7-linux -emit-llvm -o - %s | FileCheck %s --check-prefix=ARM
// RUN: %clang_cc1 -DINFRONT -Werror -triple powerpc64le-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DINFRONT -Werror -triple aarch64-linux -emit-llvm -o - %s | FileCheck %s

#ifdef INFRONT
typedef union __attribute__((transparent_union)) {
  void *f0;
} transp_t0;
#else
typedef union {
  void *f0;
} transp_t0 __attribute__((transparent_union));
#endif

void f0(transp_t0 obj);

// CHECK-LABEL: define{{.*}} void @f1_0(ptr noundef %a0)
// CHECK:  call void @f0(ptr %{{.*}})
// CHECK:  call void %{{.*}}(ptr noundef %{{[a-z0-9]*}})
// CHECK: }

// ARM-LABEL: define{{.*}} arm_aapcscc void @f1_0(ptr noundef %a0)
// ARM:  call arm_aapcscc void @f0(ptr %{{.*}})
// ARM:  call arm_aapcscc void %{{.*}}(ptr noundef %{{[a-z0-9]*}})
// ARM: }
void f1_0(int *a0) {
  void (*f0p)(void *) = f0;
  f0(a0);
  f0p(a0);
}

void f1_1(int *a0) {
  f0((transp_t0) { a0 });
}
