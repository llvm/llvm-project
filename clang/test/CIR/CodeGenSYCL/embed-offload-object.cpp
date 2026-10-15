// REQUIRES: x86-registered-target

// Verify that on the ClangIR path -fembed-offload-object embeds the packaged
// device object into the ".llvm.offloading" section of the host module,
// matching classic CodeGen. This is the default relocatable device code flow
// for SYCL.
// RUN: echo -n 'FAKE_OFFLOAD_OBJECT' > %t.out
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc \
// RUN:   -fclangir -fembed-offload-object=%t.out -emit-llvm %s -o - \
// RUN:   | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc \
// RUN:   -fembed-offload-object=%t.out -emit-llvm %s -o - \
// RUN:   | FileCheck %s

// Object emission must produce the ".llvm.offloading" section.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc \
// RUN:   -fclangir -fembed-offload-object=%t.out -emit-obj %s -o %t.o
// RUN: llvm-readelf -S %t.o | FileCheck %s --check-prefix=OBJ

// Without the flag nothing is embedded.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host -fgpu-rdc \
// RUN:   -fclangir -emit-llvm %s -o - \
// RUN:   | FileCheck %s --check-prefix=NONE --implicit-check-not='.llvm.offloading'

// Embedding does not depend on an offloading language model, as in classic
// CodeGen.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fclangir \
// RUN:   -fembed-offload-object=%t.out -emit-llvm /dev/null -o - \
// RUN:   | FileCheck %s

// A missing object file must be diagnosed.
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -fgpu-rdc -fclangir -fembed-offload-object=%t.does-not-exist \
// RUN:   -emit-llvm %s -o - 2>&1 | FileCheck %s --check-prefix=ERROR

// Embedding for CUDA, HIP and OpenMP offloading is not implemented on the
// ClangIR path yet, including when OpenMP offloading is combined with SYCL.
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -fclangir \
// RUN:   -fembed-offload-object=%t.out -emit-llvm /dev/null -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NYI
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fclangir \
// RUN:   -fembed-offload-object=%t.out -emit-llvm /dev/null -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NYI
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x c -fopenmp \
// RUN:   -fopenmp-targets=amdgcn-amd-amdhsa -fclangir \
// RUN:   -fembed-offload-object=%t.out -emit-llvm /dev/null -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NYI
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -fsycl-is-host \
// RUN:   -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -fclangir \
// RUN:   -fembed-offload-object=%t.out -emit-llvm /dev/null -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NYI

template <typename KN, typename... Ts>
void sycl_kernel_launch(const char *, Ts...) {}

template <typename KN, typename KT>
[[clang::sycl_kernel_entry_point(KN)]] void kernel_entry(KT k) { k(); }

struct KN;

int main() { kernel_entry<KN>([] {}); }

// CHECK: @llvm.embedded.object = private constant [19 x i8] c"FAKE_OFFLOAD_OBJECT", section ".llvm.offloading", align 8, !exclude
// CHECK: @llvm.compiler.used = appending global [1 x ptr] [ptr @llvm.embedded.object], section "llvm.metadata"
// CHECK: !llvm.embedded.objects = !{![[EMB:[0-9]+]]}
// CHECK: ![[EMB]] = !{ptr @llvm.embedded.object, !".llvm.offloading"}

// OBJ: .llvm.offloading

// NONE: define {{.*}}@main

// ERROR: error: could not open '{{.*}}.does-not-exist' for embedding

// NYI: error: ClangIR code gen Not Yet Implemented: embedding offload objects for CUDA, HIP or OpenMP offloading
