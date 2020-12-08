// Fails with debug checks: https://bugs.llvm.org/show_bug.cgi?id=46862
// XFAIL: !compiler-rt-optimized && !riscv64

/// Not using private alias or enabling ODR indicator can detect ODR issues.
// RUN: %clangxx_asan -fno-rtti -DBUILD_SO1 -fPIC -shared -mllvm -asan-use-private-alias=0 %s -o %dynamiclib1
// RUN: %clangxx_asan -fno-rtti -DBUILD_SO2 -fPIC -shared -mllvm -asan-use-private-alias=0 %s -o %dynamiclib2
// RUN: %clangxx_asan -fno-rtti %s %ld_flags_rpath_exe1 %ld_flags_rpath_exe2 -o %t
// RUN: %env_asan_opts=fast_unwind_on_malloc=0:detect_odr_violation=2 not %run %t 2>&1 | FileCheck %s

// RUN: %clangxx_asan -fno-rtti -DBUILD_SO1 -fPIC -shared -mllvm -asan-use-odr-indicator=1 %s -o %dynamiclib1
// RUN: %clangxx_asan -fno-rtti -DBUILD_SO2 -fPIC -shared -mllvm -asan-use-odr-indicator=1 %s -o %dynamiclib2
// RUN: %env_asan_opts=fast_unwind_on_malloc=0:detect_odr_violation=2 not %run %t 2>&1 | FileCheck %s

/// By default we can detect ODR issues in vtables.
// RUN: %clangxx_asan -fno-rtti -DBUILD_SO1 -fPIC -shared %s -o %dynamiclib1
// RUN: %clangxx_asan -fno-rtti -DBUILD_SO2 -fPIC -shared %s -o %dynamiclib2
// RUN: %env_asan_opts=fast_unwind_on_malloc=0:detect_odr_violation=2 not %run %t 2>&1 | FileCheck %s

struct XYZ {
  virtual void foo();
};

#if defined(BUILD_SO1)

void XYZ::foo() {}

#elif defined(BUILD_SO2)

void XYZ::foo() {}

#else

int main() {}

#endif

// CHECK: AddressSanitizer: odr-violation
// CHECK-NEXT: 'vtable for XYZ'
// CHECK-NEXT: 'vtable for XYZ'
