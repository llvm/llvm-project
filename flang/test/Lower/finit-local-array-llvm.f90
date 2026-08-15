! Tests that -finit-local= with non-zero patterns produces correct LLVM IR for
! static arrays and arrays inside derived types. Previously, fir.insert_on_range
! with a non-zero element failed at LLVM lowering because llvm.mlir.constant
! does not accept ArrayAttr of non-zero scalars. The fix uses a do_loop +
! coordinate_of instead, which lowers correctly through to LLVM IR.
!
! RUN: %flang_fc1 -emit-llvm -finit-local=0xAA %s -o - | FileCheck --check-prefix=HEX %s
! RUN: %flang_fc1 -emit-llvm -finit-local=nan  %s -o - | FileCheck --check-prefix=NAN %s
! RUN: %flang_fc1 -emit-llvm -finit-local=zero %s -o - | FileCheck --check-prefix=ZERO %s

! ---------------------------------------------------------------------------
! Static 1-D array INTEGER(4)(4)
! ---------------------------------------------------------------------------
subroutine test_int_array(res)
  integer(4) :: res(4)
  integer(4) :: x(4)
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! HEX:  store i32 -1431655766,
! HEX-NOT: store i32 0,

! NAN-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_int_array{{.*}}(
! ZERO: store [4 x i32] zeroinitializer,

! ---------------------------------------------------------------------------
! Derived type with an array-valued field  (Thread 2 regression)
! type t; integer :: a(2); end type; type(t) :: x
! ---------------------------------------------------------------------------
subroutine test_array_in_struct(res)
  type :: t
    integer(4) :: a(2)
  end type
  type(t) :: res
  type(t) :: x
  res = x
end subroutine
! HEX-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! HEX:  store i32 -1431655766,

! NAN-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! NAN:  store i32 -1431655766,

! ZERO-LABEL: define {{.*}}@{{.*}}test_array_in_struct{{.*}}(
! ZERO: store {{.*}} zeroinitializer,
