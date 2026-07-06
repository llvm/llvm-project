/// Tests operator new/delete mangling reflects the adapted SizeType.
/// Itanium mangling encodes the type name: _Znwm (unsigned long) vs _Znwy
/// (unsigned long long), even though both are 64-bit.

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=WIN %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=LINUX %s

typedef __SIZE_TYPE__ size_t;

[[clang::sycl_external]] void *operator new(size_t n);
[[clang::sycl_external]] void operator delete(void *p, size_t n) noexcept;

[[clang::sycl_external]] void foo() {
  int *p = new int;
  delete p;
}

// WIN: declare {{.*}} @_Znwy(i64 noundef)
// WIN: declare {{.*}} @_ZdlPvy(ptr addrspace(4) noundef, i64 noundef)
// LINUX: declare {{.*}} @_Znwm(i64 noundef)
// LINUX: declare {{.*}} @_ZdlPvm(ptr addrspace(4) noundef, i64 noundef)
