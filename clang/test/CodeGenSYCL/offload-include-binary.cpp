// Verify that -foffload-include-binary embeds the finalized SYCL device
// image into the host module and emits the registration/unregistration
// constructors and destructors expected by the SYCL runtime.
// The image is already finalized, so it must not land in ".llvm.offloading".
// RUN: echo -n 'FAKE_SYCL_DEVICE_IMAGE' > %t.bin
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.bin -emit-llvm %s -o - \
// RUN:   | FileCheck %s --implicit-check-not='.llvm.offloading' \
// RUN:     --implicit-check-not='llvm.global_ctors.'

// The registration functions have to merge into the constructor and destructor
// lists the rest of the translation unit contributes to, so object emission
// must succeed for a translation unit that has its own static initializers.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.bin -emit-obj %s -o %t.o

// Without the flag no registration IR should be emitted.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE

// A missing binary file must be diagnosed.
// RUN: not %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -foffload-include-binary %t.does-not-exist -emit-llvm %s -o - 2>&1 \
// RUN:   | FileCheck %s --check-prefix=ERROR

struct S {
  S();
  ~S();
};
S s;

void f() {}

// CHECK: @.sycl_offloading.binary = internal unnamed_addr constant [22 x i8] c"FAKE_SYCL_DEVICE_IMAGE", section ".sycl_fatbin"
// CHECK:      @llvm.global_ctors = appending global [2 x { i32, ptr, ptr }]
// CHECK-SAME:   i32 65535, ptr @_GLOBAL__sub_I_
// CHECK-SAME:   i32 1, ptr @sycl.descriptor_reg
// CHECK:      @llvm.global_dtors = {{.*}}@sycl.descriptor_unreg
// CHECK: define internal void @sycl.descriptor_reg()
// CHECK: call void @__sycl_register_lib(ptr @.sycl_offloading.binary, i64 22)
// CHECK: define internal void @sycl.descriptor_unreg()
// CHECK: call void @__sycl_unregister_lib(ptr @.sycl_offloading.binary, i64 22)

// NONE-NOT: .sycl_offloading.binary
// NONE-NOT: __sycl_register_lib

// ERROR: cannot open file '{{.*}}.does-not-exist'
