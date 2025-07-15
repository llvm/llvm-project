// Test that SYCL kernel name conflicts that occur across PCH boundaries are
// properly diagnosed.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -emit-pch -x c++-header \
// RUN:   %t/pch.h -o %t/pch.h.host.pch
// RUN: %clang_cc1 -std=c++17 -fsycl-is-host -verify \
// RUN:   -include-pch %t/pch.h.host.pch %t/test.cpp
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -emit-pch -x c++-header \
// RUN:   %t/pch.h -o %t/pch.h.device.pch
// RUN: %clang_cc1 -std=c++17 -fsycl-is-device -verify \
// RUN:   -include-pch %t/pch.h.device.pch %t/test.cpp

#--- pch.h
template<int> struct KN;

[[clang::sycl_kernel_entry_point(KN<1>)]]
void pch_test1() {} // << expected previous declaration note here.

template<typename T>
[[clang::sycl_kernel_entry_point(T)]]
void pch_test2() {} // << expected previous declaration note here.
template void pch_test2<KN<2>>();


#--- test.cpp
// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@pch.h:4 {{previous declaration is here}}
[[clang::sycl_kernel_entry_point(KN<1>)]]
void test1() {}

// expected-error@+3 {{'sycl_kernel_entry_point' kernel name argument conflicts with a previous declaration}}
// expected-note@pch.h:8 {{previous declaration is here}}
[[clang::sycl_kernel_entry_point(KN<2>)]]
void test2() {}
