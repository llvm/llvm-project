// FIXME: Same as test/asan/TestCases/Linux/odr-violation.cc ?
// XFAIL: android

// RUN: %clangxx_asan -fno-rtti -DBUILD_SO1 -fPIC -shared %s -o %t1.so
// RUN: %clangxx_asan -fno-rtti -DBUILD_SO2 -fPIC -shared %s -o %t2.so
// RUN: %clangxx_asan -fno-rtti %t1.so %t2.so %s -Wl,-R. -o %t
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
