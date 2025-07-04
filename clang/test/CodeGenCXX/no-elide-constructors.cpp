// RUN: %clang_cc1 -std=c++98 -triple i386-unknown-unknown -fno-elide-constructors -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX98
// RUN: %clang_cc1 -std=c++11 -triple i386-unknown-unknown -fno-elide-constructors -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX11
// RUN: %clang_cc1 -std=c++11 -triple amdgcn-amd-amdhsa -fno-elide-constructors -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK --check-prefix=CHECK-CXX11-NONZEROALLOCAAS
// RUN: %clang_cc1 -std=c++98 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX98-ELIDE
// RUN: %clang_cc1 -std=c++11 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX11-ELIDE
// RUN: %clang_cc1 -std=c++11 -triple amdgcn-amd-amdhsa -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX11-NONZEROALLOCAAS-ELIDE

// Reduced from PR12208
class X {
public:
  X();
  X(const X&);
#if __cplusplus >= 201103L
  X(X&&);
#endif
  ~X();
};

// CHECK-LABEL: define{{.*}} void @_Z4Testv(
// CHECK-SAME: ptr {{.*}}dead_on_unwind noalias writable sret([[CLASS_X:%.*]]) align 1 [[AGG_RESULT:%.*]])
X Test()
{
  X x;

  // Check that the copy constructor for X is called with result variable as
  // sret argument.
  // CHECK-CXX98: call void @_ZN1XC1ERKS_(
  // CHECK-CXX11: call void @_ZN1XC1EOS_(
  // CHECK-CXX11-NONZEROALLOCAAS: [[TMP0:%.*]] = addrspacecast ptr addrspace(5) [[AGG_RESULT]] to ptr
  // CHECK-CXX11-NONZEROALLOCAAS-NEXT: call void @_ZN1XC1EOS_(ptr noundef nonnull align 1 dereferenceable(1) [[TMP0]]
  // CHECK-CXX98-ELIDE-NOT: call void @_ZN1XC1ERKS_(
  // CHECK-CXX11-ELIDE-NOT: call void @_ZN1XC1EOS_(
  // CHECK-CXX11-NONZEROALLOCAAS-ELIDE-NOT: call void @_ZN1XC1EOS_(

  // Make sure that the destructor for X is called.
  // FIXME: This call is present even in the -ELIDE runs, but is guarded by a
  // branch that is never taken in those cases. We could generate better IR
  // here.
  // CHECK: call void @_ZN1XD1Ev(
  return x;
}
