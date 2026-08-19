// REQUIRES: x86-registered-target

// Verify that -foffload-include-binary embeds the finalized SYCL device
// binary into the host module and emits the registration/unregistration
// constructors and destructors expected by the SYCL runtime.
// The binary is already finalized, so it must not land in ".llvm.offloading".
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

// Without the flag no SYCL registration IR should be emitted.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsycl-is-host \
// RUN:   -emit-llvm %s -o - | FileCheck %s --check-prefix=NONE \
// RUN:     --implicit-check-not='.sycl_offloading.binary' \
// RUN:     --implicit-check-not='__sycl_register_lib' \
// RUN:     --implicit-check-not='llvm.global_dtors'

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
// CHECK:      define internal void @sycl.descriptor_reg()
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__sycl_register_lib(ptr @.sycl_offloading.binary, i64 22)
// CHECK-NEXT:   ret void
// CHECK:      define internal void @sycl.descriptor_unreg()
// CHECK-NEXT: entry:
// CHECK-NEXT:   call void @__sycl_unregister_lib(ptr @.sycl_offloading.binary, i64 22)
// CHECK-NEXT:   ret void

// NONE:      @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }]
// NONE-SAME:   i32 65535, ptr @_GLOBAL__sub_I_
// NONE:      define dso_local void @_Z1fv()

// ERROR: cannot open file '{{.*}}.does-not-exist'
