// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck -check-prefix=CHECK-TAPI %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | \
// RUN: FileCheck -check-prefix=CHECK-TAPI2 %s

// RUN: %clang -target x86_64-unknown-linux-gnu -c -o - %s | llvm-readelf -s - 2>&1 | \
// RUN: FileCheck -check-prefix=CHECK-SYMBOLS %s

// For the following:
// g()
// n::S<int>::S()
// n::S<int>::~S()
// n::S<int>::func() const
// n::S<int>::S(n::S<int> const&)

// We expect these manglings:
// CHECK-TAPI: Symbols:
// CHECK-TAPI-NOT: _ZNK1n1SIiEclEv
// CHECK-TAPI2: Symbols:
// CHECK-TAPI2: _Z1g

// CHECK-SYMBOLS-DAG: FUNC    GLOBAL DEFAULT    {{[0-9]}} _Z1g
// CHECK-SYMBOLS-DAG: FUNC    WEAK   HIDDEN     {{[0-9]}} _ZNK1n1SIiEclEv

namespace n {
template <typename T>
struct __attribute__((__visibility__("default"))) S {
  S() = default;
  ~S() = default;
  int __attribute__((__visibility__(("default")))) func() const {
    return 1844;
  }
  int __attribute__((__visibility__(("hidden")))) operator()() const {
    return 1863;
  }
};
} // namespace n

void g() { n::S<int>()(); }
