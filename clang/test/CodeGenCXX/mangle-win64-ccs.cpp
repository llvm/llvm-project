// RUN: %clang_cc1 -triple x86_64-windows-gnu -o - -emit-llvm %s | FileCheck %s -check-prefix CHECK-WIN
// RUN: %clang_cc1 -triple x86_64-linux-gnu -o - -emit-llvm %s | FileCheck %s -check-prefix CHECK-LIN

typedef __PTRDIFF_TYPE__ ptrdiff_t;
template <typename FTy> ptrdiff_t func_as_int(FTy *fp) { return ptrdiff_t(fp); }

int f_plain(int);
int __attribute__((sysv_abi)) f_sysvabi(int);
int __attribute__((ms_abi)) f_msabi(int);
ptrdiff_t useThem() {
  ptrdiff_t rv = 0;
  rv += func_as_int(f_plain);
  rv += func_as_int(f_sysvabi);
  rv += func_as_int(f_msabi);
  return rv;
}

// CHECK-WIN: define dso_local i64 @_Z7useThemv()
// CHECK-WIN:   call i64 @_Z11func_as_intIFiiEExPT_(i32 (i32)* @_Z7f_plaini)
// CHECK-WIN:   call i64 @_Z11func_as_intIU8sysv_abiFiiEExPT_(i32 (i32)* @_Z9f_sysvabii)
// CHECK-WIN:   call i64 @_Z11func_as_intIFiiEExPT_(i32 (i32)* @_Z7f_msabii)

// CHECK-LIN: define{{.*}} i64 @_Z7useThemv()
// CHECK-LIN:   call i64 @_Z11func_as_intIFiiEElPT_(i32 (i32)* @_Z7f_plaini)
// CHECK-LIN:   call i64 @_Z11func_as_intIFiiEElPT_(i32 (i32)* @_Z9f_sysvabii)
// CHECK-LIN:   call i64 @_Z11func_as_intIU6ms_abiFiiEElPT_(i32 (i32)* @_Z7f_msabii)
