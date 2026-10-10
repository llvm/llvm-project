// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -verify -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -verify -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -fsycl-is-device -triple spir64 -verify -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// expected-no-diagnostics

template <typename Name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

// Function pointers stay in the program address space (0); data pointers use
// the generic address space (4).
// CIR: cir.func {{.*}}@_Z15invoke_functionPFivEPi(%{{.*}}: !cir.ptr<!cir.func<() -> !s32i>> {{.*}}, %{{.*}}: !cir.ptr<!s32i, target_address_space(4)> {{.*}}) cc(spir_function)
// LLVM: define dso_local spir_func{{.*}}invoke_function{{.*}}(ptr noundef %{{.*}}, ptr addrspace(4) noundef %{{.*}})
[[clang::sycl_external]] void invoke_function(int (*fptr)(), int *ptr) {}

int f() { return 0; }

int main() {
  kernel_single_task<class fake_kernel>([=]() {
    int (*p)() = f;
    int (&r)() = *p;
    int a = 10;
    invoke_function(p, &a);
    invoke_function(r, &a);
    invoke_function(f, &a);
  });
  return 0;
}
